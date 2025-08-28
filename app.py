# ========================== app.py ============================
import os
import uuid
import time
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from dotenv import load_dotenv
from pydub import AudioSegment
from faster_whisper import WhisperModel
import smtplib
from email.mime.text import MIMEText

# Existing utilities and DB functions
from scripts.utils.parser import extract_text_from_pdf
from scripts.utils.embedder import embed_resume
from scripts.agent import (
    generate_25_questions,
    handle_candidate_text_answer_fast,  # must return dict with keys: score, explanation (or None)
    warm_llm,
)
from scripts.db import (
    init_db, create_candidate, save_answer, finish_candidate,
    get_candidate, get_leaderboard, get_candidate_answers, update_candidate_score
)
from scripts.org import require_org_auth

# ------------------------- CONFIG -------------------------
# Keep TF knobs out of the way if present in some deps
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

for d in ["data/candidate", "data/org", "vectorstore/candidate", "vectorstore/org", "audio", "static/videos", "templates"]:
    os.makedirs(d, exist_ok=True)

# Init DB and warm local LLM (if your agent uses it)
init_db("answers.db")
warm_llm()

# Whisper Model (CPU-friendly default)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

# ------------------------- HELPERS -------------------------
def transcribe_audio_whisper(path: str) -> str:
    # Faster-whisper returns segments; we stitch text together
    segments, _ = whisper_model.transcribe(path, beam_size=1)
    return " ".join([seg.text for seg in segments]).strip()


# ------------------------- ROUTES -------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/candidate", methods=["GET", "POST"])
def candidate_upload_page():
    if request.method == "GET":
        return render_template("candidate_upload.html")

    file = request.files.get("resume")
    if not file or not file.filename.lower().endswith(".pdf"):
        return render_template("candidate_upload.html", error="Upload a valid PDF file (.pdf).")

    # Optional email (no validation required)
    email = (request.form.get("email") or "").strip() or None

    # Create candidate id and session
    cid = str(uuid.uuid4())
    session["candidate_id"] = cid

    # Save resume
    c_dir = os.path.join("data/candidate", cid)
    os.makedirs(c_dir, exist_ok=True)
    pdf_path = os.path.join(c_dir, "resume.pdf")
    file.save(pdf_path)

    # Extract text & embed
    try:
        resume_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        return render_template("candidate_upload.html", error=f"Failed to read PDF: {e}")

    try:
        embed_resume(resume_text, user_id=cid, role="candidate")
    except Exception as e:
        # Non-fatal: you can still proceed with static questions
        print(f"[WARN] Embedding resume failed: {e}")

    # Create candidate record
    create_candidate(cid, email=email)

    # Generate questions
    try:
        questions = generate_25_questions(cid, resume_text)
    except Exception as e:
        print(f"[WARN] generate_25_questions failed: {e}")
        questions = [
            "Tell me about your most impactful project.",
            "Explain the concept of containers and Docker.",
            "What is your approach to debugging production issues?"
        ]

    session["questions"] = questions
    session["q_index"] = 0

    # Ready page with "Start Interview" button
    return render_template("candidate_ready.html")


@app.route("/candidate/interview", methods=["GET"])
def candidate_interview():
    if not session.get("candidate_id"):
        return redirect(url_for("candidate_upload_page"))
    return render_template("candidate_interview.html")


@app.route("/candidate/next_question", methods=["GET"])
def candidate_next_question():
    cid = session.get("candidate_id")
    if not cid:
        return jsonify({"done": True, "message": "Session expired."})

    q_index = session.get("q_index", 0)
    questions = session.get("questions", [])

    if q_index >= len(questions):
        finish_candidate(cid)
        update_candidate_score(cid)
        return jsonify({"done": True, "message": "Interview complete!"})

    question = questions[q_index]
    # We do NOT increment here; we increment only after we receive the answer
    return jsonify({"done": False, "question": question, "q_index": q_index + 1})


@app.route("/candidate/answer", methods=["POST"])
def candidate_answer_audio():
    cid = session.get("candidate_id")
    if not cid:
        return jsonify({"error": "Session expired."}), 440  # 440 Login Time-out (semantically)

    audio = request.files.get("audio")
    if not audio:
        return jsonify({"error": "No audio received"}), 400

    # Save audio
    c_dir = os.path.join("audio", cid)
    os.makedirs(c_dir, exist_ok=True)
    ts = int(time.time())
    raw_path = os.path.join(c_dir, f"answer_raw_{ts}.webm")
    wav_path = os.path.join(c_dir, f"answer_{ts}.wav")
    audio.save(raw_path)

    # Convert to wav (mono/16k) for whisper
    try:
        sound = AudioSegment.from_file(raw_path)  # requires ffmpeg on PATH
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {e}. Ensure ffmpeg is installed and on PATH."}), 500
    finally:
        try:
            os.remove(raw_path)
        except Exception:
            pass

    # Get current question
    q_index = session.get("q_index", 0)
    questions = session.get("questions", [])
    if q_index >= len(questions):
        finish_candidate(cid)
        update_candidate_score(cid)
        return jsonify({"done": True, "message": "Interview already complete!"})

    current_question = questions[q_index]

    # Transcribe audio
    try:
        transcript = transcribe_audio_whisper(wav_path)
    except Exception as e:
        transcript = ""
        print(f"[WARN] Transcription failed: {e}")

    # Ask agent to (quickly) decide/store scoring info
    try:
        # IMPORTANT: handle_candidate_text_answer_fast should return a dict
        # with keys: score (float or None), explanation (str or None)
        result = handle_candidate_text_answer_fast(
            candidate_id=cid,
            question=current_question,
            transcript=transcript
        )
        # Backward-compat: if a float is returned, wrap it
        if isinstance(result, (int, float)):
            result = {"score": float(result), "explanation": None}
        elif not isinstance(result, dict):
            result = {"score": 0.0, "explanation": "Unexpected scoring response"}
        score = float(result.get("score") or 0.0)
        explanation = result.get("explanation")
    except Exception as e:
        print(f"[WARN] Scoring failed: {e}")
        score, explanation = 0.0, f"Scoring failed: {e}"

    # Save in DB
    save_answer(cid, current_question, transcript, score, explanation)

    # Increment question index AFTER saving the answer
    session["q_index"] = q_index + 1
    done = session["q_index"] >= len(questions)
    if done:
        finish_candidate(cid)
        update_candidate_score(cid)

    return jsonify({
        "done": done,
        "transcript": transcript,
        "score": score,
        "explanation": explanation or ""
    })


@app.route("/candidate/finish", methods=["GET"])
def candidate_finish_page():
    cid = session.get("candidate_id")
    if not cid:
        return redirect(url_for("candidate_upload_page"))

    cand = get_candidate(cid)
    answers = get_candidate_answers(cid)

    # Scores
    interview_score = sum([a["score"] for a in answers]) / len(answers) if answers else 0
    resume_score = 50  # Static for now
    total_score = round(resume_score * 0.4 + interview_score * 0.6, 2)

    return render_template("candidate_finished.html", candidate=cand, answers=answers,
                           resume_score=resume_score, interview_score=interview_score,
                           total_score=total_score)


@app.route("/leaderboard", methods=["GET", "POST"])
def leaderboard_view():
    if request.method == "POST":
        selected_candidates = request.form.getlist("selected_candidates")
        for cid in selected_candidates:
            cand = get_candidate(cid)
            if cand and cand.get("email"):
                send_email(cand["email"], "You are shortlisted", f"Dear Candidate,\n\nCongratulations! You are shortlisted.")
        return redirect(url_for("leaderboard_view"))

    rows = get_leaderboard()
    return render_template("leaderboard.html", rows=rows)


@app.route("/org/login", methods=["GET", "POST"])
def org_login():
    if request.method == "POST":
        email = request.form.get("email")
        pwd = request.form.get("password")
        if email == "admin@org.com" and pwd == "admin123":
            session["org_authed"] = True
            return redirect(url_for("org_dashboard"))
        return render_template("org_login.html", error="Invalid credentials")
    return render_template("org_login.html")


@app.route("/org/dashboard", methods=["GET"])
@require_org_auth
def org_dashboard():
    rows = get_leaderboard()
    return render_template("org_dashboard.html", rows=rows)


@app.route("/org/candidate/<cid>", methods=["GET"])
@require_org_auth
def org_view_candidate(cid):
    cand = get_candidate(cid)
    answers = get_candidate_answers(cid)
    if not cand:
        return "Not found", 404
    return render_template("org_candidate_view.html", candidate=cand, answers=answers)

# ------------------------- EMAIL UTILITY -------------------------
def send_email(to_email, subject, body):
    sender_email = os.getenv("ORG_EMAIL")
    password = os.getenv("ORG_EMAIL_PASSWORD")
    if not sender_email or not password:
        print("Email not configured.")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, to_email, msg.as_string())
    except Exception as e:
        print(f"Email failed: {e}")


if __name__ == "__main__":
    # You can also run via waitress; for local dev, Flask is fine
    app.run(host="127.0.0.1", port=5000, debug=False)
