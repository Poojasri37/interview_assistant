import os
import random
from typing import List

import google.generativeai as genai


def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
    genai.configure(api_key=api_key)
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name)


def _call_gemini(prompt: str) -> str:
    model = _get_gemini_model()
    resp = model.generate_content(prompt)
    # google-generativeai >= 0.5.0 exposes .text
    return resp.text


def _parse_questions(text: str, num_questions: int) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    questions: List[str] = []

    for ln in lines:
        # Remove bullets
        q = ln.lstrip("-*•").strip()
        # Remove numbering like "1. ", "2) ", "3 - "
        while q and (q[0].isdigit() or q[0] in ".-)"):
            q = q[1:]
        q = q.strip()
        if q:
            questions.append(q)

    # Deduplicate, keep order
    seen = set()
    uniq: List[str] = []
    for q in questions:
        if q not in seen:
            seen.add(q)
            uniq.append(q)

    return uniq[:num_questions]


def generate_questions_with_gemini(
    candidate_id: str,
    resume_text: str,
    num_questions: int = 10
) -> List[str]:
    """
    Use Gemini to generate domain-specific interview questions based on the candidate's resume.
    Hybrid idea:
      - Prompt is grounded on actual resume text (projects, skills, domain)
      - Local RAG (FAISS) is still used later in agent.py for explanations/feedback
    """
    # Trim resume so prompt is not too long
    snippet = resume_text
    max_chars = 4000
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars]

    prompt = f"""
You are an AI interview assistant.

Below is a candidate's resume content:

\"\"\"{snippet}\"\"\"

From this resume, infer:
- Core technical skills
- Main domain (e.g., Computer Science, ECE, Mechanical, Civil, EEE, etc.)
- Key projects and responsibilities

Now generate {num_questions} domain-specific, technical interview questions that:
- Directly test the candidate's skills and projects mentioned in the resume
- Start from basic questions and gradually become more challenging
- Are concise and clear
- Do NOT include the answers
- Do NOT add extra commentary

Return ONLY the questions, one per line. No numbering explanation text.
    """

    try:
        raw = _call_gemini(prompt)
        questions = _parse_questions(raw, num_questions)
        if not questions:
            raise ValueError("Gemini returned no usable questions.")
        return questions
    except Exception as e:
        print(f"[WARN] generate_questions_with_gemini fallback due to: {e}")
        # Fallback: static but reasonable questions
        fallback = [
            "Tell me about the most complex project you have worked on.",
            "Explain a technical challenge you solved and how you approached it.",
            "Describe how you applied your core engineering knowledge in a real-world scenario.",
            "How do you debug and fix issues in your projects?",
            "Explain a tool or technology you frequently use and why.",
            "Describe a time you worked in a team to deliver a project.",
            "How do you stay updated in your field?",
            "Explain a concept from your core domain that you find interesting.",
            "Describe an optimization or improvement you implemented in any project.",
            "Tell me about a time when your project did not go as planned and what you did."
        ]
        random.shuffle(fallback)
        return fallback[:num_questions]
