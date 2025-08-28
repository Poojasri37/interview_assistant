import re
from typing import Any

_NUMBER_RE = re.compile(r'(?<!\d)(10(?:\.0+)?|\d(?:\.\d+)?)(?!\d)')

def _call_llm(llm: Any, prompt: str) -> str:
    if hasattr(llm, "invoke"):
        out = llm.invoke(prompt)
        return out if isinstance(out, str) else str(out)
    if hasattr(llm, "predict"):
        return llm.predict(prompt)
    try:
        out = llm(prompt)
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            return out[0].get("generated_text", str(out))
        return str(out)
    except Exception:
        return ""

def _extract_score(text: str) -> float | None:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    try:
        val = float(m.group(1))
        if 0.0 <= val <= 10.0:
            return val
    except Exception:
        pass
    return None

def score_answer_with_llm(llm, question: str, answer: str, candidate_id: str | None = None, retries: int = 1) -> float:
    prompt = f"""You are an interviewer. Score the candidate's answer strictly from 0 to 10.
Consider: correctness, relevance, depth, clarity.

Return ONLY the number. No words, no explanation.

Question: {question}
Answer: {answer}

Score:"""

    last_text = ""
    for _ in range(retries + 1):
        text = _call_llm(llm, prompt)
        last_text = text
        score = _extract_score(text)
        if score is not None:
            return round(score, 2)

    # fallback heuristic
    if not answer.strip():
        return 0.0
    length_factor = min(len(answer.split()) / 80.0, 1.0)
    depth_bonus = 2.0 if any(k in answer.lower() for k in ["because", "architecture", "scalable", "tradeoff"]) else 0.0
    heuristic = max(0.0, min(10.0, 4.0 + length_factor * 4.0 + depth_bonus))
    return round(heuristic, 2)
