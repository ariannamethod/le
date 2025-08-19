import sqlite3
import hashlib
from typing import Optional


class Memory:
    """Simple SQLite-backed storage for meta information and conversations."""

    def __init__(self, path: str = "memory.db") -> None:
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT)"
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT
            )
            """
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def save_conversation(self, question: str, answer: str) -> None:
        self.conn.execute(
            "INSERT INTO conversations(question, answer) VALUES (?, ?)",
            (question, answer),
        )
        self.conn.commit()

    @staticmethod
    def hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
