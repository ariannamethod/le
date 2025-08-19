import hashlib
import logging
import sqlite3
from pathlib import Path
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

    # New functionality for message tracking and repository hashing

    def record_message(self, question: str, answer: str) -> None:
        """Persist a conversation pair."""
        self.save_conversation(question, answer)

    def update_repo_hash(self, repo_path: str | Path = ".") -> None:
        """Compute SHA256 for every file and flag training if anything changed."""
        repo = Path(repo_path)
        changed = False
        db_path = Path(self.conn.execute("PRAGMA database_list").fetchone()[2])
        for file in repo.rglob("*"):
            if (
                file.is_file()
                and ".git" not in file.parts
                and file != db_path
            ):
                digest = self.hash_file(str(file))
                key = f"hash:{file.relative_to(repo)}"
                if self.get_meta(key) != digest:
                    logging.info("Hash for %s changed", file)
                    self.set_meta(key, digest)
                    changed = True
        if changed:
            self.set_meta("needs_training", "1")

    def needs_training(self) -> bool:
        """Return True if retraining is required."""
        return self.get_meta("needs_training") == "1"

    @staticmethod
    def hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
