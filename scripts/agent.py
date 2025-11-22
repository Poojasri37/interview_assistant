import os
import random
import threading
from typing import Callable, Optional, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from scripts.utils.scoring import score_answer_with_llm

# -------------------- CONFIG --------------------
MODEL_ID = os.getenv("LOCAL_LLM_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("RAG_TOP_K", "3"))

_LLM: Optional[HuggingFacePipeline] = None  # cached HF pipeline

EXPLAIN_TRIGGERS = {
    "explain",
    "explain please",
    "can you explain",
    "idk",
    "i dont know",
    "i don't know",
    "no idea",
}

# -------------------- PUBLIC API (imported by app.py) --------------------
def warm_llm():
    """Warm the local LLM (so first call isn't slow)."""
    _load_llm()


def generate_25_questions(candidate_id: str, resume_text: str):
    """
    Legacy helper: static question set (not used by app.py now, which uses Gemini).
    Kept for compatibility / debugging.
    """
    base = [
        "Summarize your most impactful project.",
        "Explain the biggest technical challenge you solved.",
        "Which programming languages are you most comfortable with and why?",
        "Explain a time you optimized performance in a system you built.",
        "What cloud technologies have you used and on what projects?",
        "What is your approach to debugging complex production issues?",
        "Describe your experience with databases.",
        "Tell me about a time you led a team or initiative.",
        "How do you keep updated with the latest tech trends?",
        "Describe a situation where you automated a workflow.",
        "Explain SOLID principles (pick 1–2 that you know best).",
        "How do you design a scalable API?",
        "What is CI/CD and how have you used it?",
        "Describe your testing strategy in projects.",
        "How do you approach security in your applications?",
        "What is the difference between synchronous and asynchronous programming?",
        "Explain the architecture of one of your major projects.",
        "How do you handle failures and retries in distributed systems?",
        "Explain a data structure you frequently used and why.",
        "What is your approach to documentation?",
        "What is REST vs GraphQL?",
        "Explain the concept of containers and Docker.",
        "What version control flows have you used (e.g., Git-flow)?",
        "How do you measure success in a project?",
        "Tell me about a time you had to say 'I don't know'."
    ]
    random.shuffle(base)
    return base[:25]


def handle_candidate_text_answer_fast(
    candidate_id: str,
    question: str,
    transcript: str,
    force_next_if_empty: bool = True,
    no_response_token: str = "[NO RESPONSE]",
) -> Dict[str, Any]:
    """
    Synchronous scoring used by app.py.

    Returns:
      {
        "score": float,
        "explanation": Optional[str]
      }
    """
    transcript = (transcript or "").strip()

    # No speech captured
    if not transcript:
        if not force_next_if_empty:
            return {
                "score": 0.0,
                "explanation": "No answer captured; please repeat."
            }
        return {
            "score": 0.0,
            "explanation": "No answer detected. Try speaking a bit louder or closer to the mic."
        }

    # Candidate explicitly asks for explanation instead of answering
    if _is_explain_trigger(transcript):
        try:
            explanation = _rag_answer(
                candidate_id,
                "candidate",
                f"Explain this interview question in simple terms and give a short model answer: {question}"
            )
        except Exception as e:
            explanation = f"Explanation failed: {e}"
        # Score is 0 here because they didn't really answer
        return {
            "score": 0.0,
            "explanation": explanation
        }

    # Normal scoring path
    try:
        score = score_answer_with_llm(
            llm=_load_llm(),
            question=question,
            answer=transcript,
            candidate_id=candidate_id,
        )
    except Exception as e:
        return {
            "score": 0.0,
            "explanation": f"Scoring failed: {e}"
        }

    explanation: Optional[str] = None
    # If score is low, give RAG-based explanation/feedback using the resume vector DB
    try:
        if score is not None and float(score) < 4.0:
            explanation = _rag_answer(
                candidate_id,
                "candidate",
                f"""
You are an interview coach.
The question was: {question}
The candidate answered: {transcript}

1) Briefly explain what a strong answer should include.
2) Tell the candidate how they can improve their answer next time.
                """.strip()
            )
    except Exception as e:
        explanation = f"Could not generate detailed explanation: {e}"

    return {
        "score": float(score) if score is not None else 0.0,
        "explanation": explanation,
    }


def async_score_answer_text(
    transcript: str,
    question: str,
    candidate_id: str,
    save_func: Callable[[float, Optional[str]], None],
):
    """
    Legacy async scorer (not used by current app.py, but kept for completeness).
    """

    def _task():
        try:
            score = score_answer_with_llm(
                llm=_load_llm(),
                question=question,
                answer=transcript,
                candidate_id=candidate_id,
            )
            explanation = None
            if score < 4.0:
                explanation = _rag_answer(candidate_id, "candidate", f"Explain the question: {question}")
            save_func(score, explanation)
        except Exception as e:
            save_func(0.0, f"Scoring failed: {e}")

    threading.Thread(target=_task, daemon=True).start()


def explain_last_question(user_id: str, last_question: str) -> str:
    return _rag_answer(user_id, "candidate", f"Explain the question '{last_question}' in simple terms.")


# -------------------- INTERNALS --------------------
def _load_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )
    _LLM = HuggingFacePipeline(pipeline=gen)
    return _LLM


def _get_retriever(user_id: str, role: str):
    """
    Load FAISS vectorstore for the given user+role and return a retriever.
    This uses the embeddings created by scripts.utils.embedder.embed_resume.
    """
    path = f"vectorstore/{role}/{user_id}"
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectordb = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return vectordb.as_retriever(search_kwargs={"k": TOP_K})


def _rag_answer(user_id: str, role: str, question: str) -> str:
    """
    RAG over user's resume:
      - FAISS + sentence-transformers embeddings
      - Local LLM (TinyLlama by default)
    """
    llm = _load_llm()
    retriever = _get_retriever(user_id, role)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa.run(question)


def _is_explain_trigger(t: str) -> bool:
    t_norm = (t or "").lower().strip()
    if t_norm in EXPLAIN_TRIGGERS:
        return True
    if "explain" in t_norm and len(t_norm.split()) <= 6:
        return True
    return False
