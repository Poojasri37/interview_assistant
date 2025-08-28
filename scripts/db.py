import sqlite3
from typing import List, Optional

_DB_PATH = "answers.db"


def init_db(db_path: str):
    """Create tables if missing and migrate schema (email column)."""
    global _DB_PATH
    _DB_PATH = db_path
    conn = sqlite3.connect(_DB_PATH)
    cur = conn.cursor()

    # Base candidates table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS candidates (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        finished INTEGER DEFAULT 0,
        avg_score REAL DEFAULT 0
    )
    """)

    # Answers table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        candidate_id TEXT,
        question TEXT,
        answer TEXT,
        score REAL,
        explanation TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Migrate: add email column if missing
    cur.execute("PRAGMA table_info(candidates)")
    columns = [row[1] for row in cur.fetchall()]
    if "email" not in columns:
        cur.execute("ALTER TABLE candidates ADD COLUMN email TEXT")

    conn.commit()
    conn.close()


def _conn():
    return sqlite3.connect(_DB_PATH)


def create_candidate(cid: str, email: Optional[str] = None):
    with _conn() as conn:
        conn.execute("INSERT INTO candidates (id, email) VALUES (?, ?)", (cid, email))
        conn.commit()


def save_answer(candidate_id: str, question: str, answer: str, score: float, explanation: Optional[str] = None):
    with _conn() as conn:
        conn.execute("""
            INSERT INTO answers (candidate_id, question, answer, score, explanation)
            VALUES (?, ?, ?, ?, ?)
        """, (candidate_id, question, answer, score, explanation))
        conn.commit()


def finish_candidate(cid: str):
    with _conn() as conn:
        conn.execute("UPDATE candidates SET finished=1 WHERE id=?", (cid,))
        conn.commit()


def update_candidate_score(cid: str):
    with _conn() as conn:
        cur = conn.execute("SELECT AVG(score) FROM answers WHERE candidate_id=?", (cid,))
        avg = cur.fetchone()[0] or 0
        conn.execute("UPDATE candidates SET avg_score=? WHERE id=?", (avg, cid))
        conn.commit()


def get_candidate(cid: str):
    with _conn() as conn:
        cur = conn.execute("SELECT id, email, created_at, finished, avg_score FROM candidates WHERE id=?", (cid,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "email": row[1],
            "created_at": row[2],
            "finished": bool(row[3]),
            "avg_score": row[4]
        }


def get_candidate_answers(cid: str) -> List[dict]:
    with _conn() as conn:
        cur = conn.execute("""
            SELECT question, answer, score, explanation, created_at
            FROM answers WHERE candidate_id=?
            ORDER BY id ASC
        """, (cid,))
        rows = cur.fetchall()
        return [
            {"question": r[0], "answer": r[1], "score": r[2], "explanation": r[3], "created_at": r[4]}
            for r in rows
        ]


def list_candidates() -> List[dict]:
    with _conn() as conn:
        cur = conn.execute("SELECT id, email, created_at, finished, avg_score FROM candidates ORDER BY created_at DESC")
        rows = cur.fetchall()
        return [{"id": r[0], "email": r[1], "created_at": r[2], "finished": bool(r[3]), "avg_score": r[4]} for r in rows]


def get_leaderboard() -> List[dict]:
    with _conn() as conn:
        cur = conn.execute("""
            SELECT id, email, avg_score FROM candidates
            WHERE finished=1
            ORDER BY avg_score DESC
        """)
        return [{"candidate_id": r[0], "email": r[1], "avg_score": r[2]} for r in cur.fetchall()]
