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
    handle_candidate_text_answer_fast,  # returns dict: {"score": float, "explanation": str|None}
    warm_llm,
)
from scripts.db import (
    init_db, create_candidate, save_answer, finish_candidate,
    get_candidate, get_leaderboard, get_candidate_answers, update_candidate_score
)
from scripts.org import require_org_auth
from scripts.question_generator import generate_questions_with_gemini  # NEW

# ------------------------- CONFIG -------------------------
os.environ["USE_TF"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "change-me")

for d in ["data/candidate", "data/org", "vectorstore/candidate", "vectorstore/org", "audio", "static/videos", "templates"]:
    os.makedirs(d, exist_ok=True)

init_db("answers.db")
warm_llm()

# Whisper Model (CPU-friendly default)
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base.en")
whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

# ------------------------- HELPERS -------------------------
def transcribe_audio_whisper(path: str) -> str:
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

    # Optional email
    email = (request.form.get("email") or "").strip() or None

    cid = str(uuid.uuid4())
    session["candidate_id"] = cid

    # Save resume
    c_dir = os.path.join("data/candidate", cid)
    os.makedirs(c_dir, exist_ok=True)
    pdf_path = os.path.join(c_dir, "resume.pdf")
    file.save(pdf_path)

    # Extract text
    try:
        resume_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        return render_template("candidate_upload.html", error=f"Failed to read PDF: {e}")

    # Embed resume into vector DB for RAG
    try:
        embed_resume(resume_text, user_id=cid, role="candidate")
    except Exception as e:
        print(f"[WARN] Embedding resume failed: {e}")

    # Create candidate record
    create_candidate(cid, email=email)

    # Generate domain-specific questions using Gemini, grounded in resume
    try:
        # You can change num_questions here if you want more/less
        questions = generate_questions_with_gemini(candidate_id=cid, resume_text=resume_text, num_questions=10)
    except Exception as e:
        print(f"[WARN] Gemini question generation failed: {e}")
        # Fallback generic questions
        questions = [
            "Tell me about your most impactful project.",
            "Explain a challenging bug or problem you solved.",
            "Describe how you applied your core technical skills in a real project.",
            "Explain a time you worked in a team to deliver a project.",
            "How do you keep your skills up to date?"
        ]

    session["questions"] = questions
    session["q_index"] = 0

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
        return jsonify({"error": "Session expired."}), 440

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

    # Convert to wav
    try:
        sound = AudioSegment.from_file(raw_path)
        sound = sound.set_frame_rate(16000).set_channels(1)
        sound.export(wav_path, format="wav")
    except Exception as e:
        return jsonify({"error": f"Audio conversion failed: {e}. Ensure ffmpeg is installed and on PATH."}), 500
    finally:
        try:
            os.remove(raw_path)
        except Exception:
            pass

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

    # Score + explanation using local LLM + RAG
    try:
        result = handle_candidate_text_answer_fast(
            candidate_id=cid,
            question=current_question,
            transcript=transcript
        )
        if isinstance(result, (int, float)):
            result = {"score": float(result), "explanation": None}
        elif not isinstance(result, dict):
            result = {"score": 0.0, "explanation": "Unexpected scoring response"}
        score = float(result.get("score") or 0.0)
        explanation = result.get("explanation")
    except Exception as e:
        print(f"[WARN] Scoring failed: {e}")
        score, explanation = 0.0, f"Scoring failed: {e}"

    save_answer(cid, current_question, transcript, score, explanation)

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

    interview_score = sum([a["score"] for a in answers]) / len(answers) if answers else 0
    resume_score = 50
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
                send_email(
                    cand["email"],
                    "You are shortlisted",
                    f"Dear Candidate,\n\nCongratulations! You are shortlisted."
                )
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
    app.run(host="127.0.0.1", port=5000, debug=False)
