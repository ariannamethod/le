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

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

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
                context TEXT,
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

    def save_conversation(self, context: str, question: str, answer: str) -> None:
        self.conn.execute(
            "INSERT INTO conversations(context, question, answer) VALUES (?, ?, ?)",
            (context, question, answer),
        )
        self.conn.commit()

    # New functionality for message tracking and repository hashing

    def record_message(self, question: str, answer: str, context: str) -> None:
        """Persist a conversation pair with associated context."""
        self.save_conversation(context, question, answer)

    def get_messages(self, limit: int | None = None) -> list[str]:
        """Return recent conversation lines in chronological order."""
        cur = self.conn.cursor()
        query = "SELECT question, answer FROM conversations ORDER BY id DESC"
        if limit is not None:
            cur.execute(query + " LIMIT ?", (limit,))
        else:
            cur.execute(query)
        rows = cur.fetchall()
        lines: list[str] = []
        for question, answer in reversed(rows):
            if question:
                lines.append(question)
            if answer:
                lines.append(answer)
        return lines

    def get_conversations(self, limit: int | None = None) -> list[tuple[str, str, str]]:
        """Return stored conversations including context in chronological order."""
        cur = self.conn.cursor()
        query = "SELECT context, question, answer FROM conversations ORDER BY id"
        if limit is not None:
            cur.execute(query + " LIMIT ?", (limit,))
        else:
            cur.execute(query)
        return cur.fetchall()

    def update_repo_hash(self, repo_path: str | Path = ".") -> None:
        """Compute file hashes and flag training when source files change.

        Temporary artefacts such as logs or databases are ignored so that
        only relevant source data and code trigger retraining.
        """

        repo = Path(repo_path)
        changed = False
        db_path = Path(self.conn.execute("PRAGMA database_list").fetchone()[2])
        ignored_dirs = {".git", "logs", "__pycache__", ".pytest_cache"}
        ignored_names = {db_path.name, "memory.db"}

        for file in repo.rglob("*"):
            if not file.is_file():
                continue
            if any(part in ignored_dirs for part in file.parts):
                continue
            if file.name in ignored_names:
                continue

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
