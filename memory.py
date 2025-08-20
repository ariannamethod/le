import hashlib
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional


class Memory:
    """Simple SQLite-backed storage for meta information and conversations."""

    def __init__(self, path: str = "memory.db") -> None:
        self.conn = sqlite3.connect(path)
        self._init_db()
        # Establish a baseline of hashes so that startup does not trigger
        # training. Actual changes will be detected in subsequent calls.
        self.update_repo_hash(initial=True)

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

    def update_repo_hash(
        self, repo_path: str | Path = ".", *, initial: bool = False
    ) -> None:
        """Compute file hashes and flag training when source files change.

        Tracks all files for code changes and specifically watches the
        ``blood/`` and ``datasets/`` directories. When data files change we
        accumulate their sizes and trigger training once the total exceeds
        ``LE_TRAINING_LIMIT_BYTES`` (default 5KB).

        Temporary artefacts such as logs or databases are ignored so that
        only relevant source data and code trigger retraining.
        """

        repo = Path(repo_path)
        code_changed = False
        data_changed_bytes = 0

        db_path = Path(self.conn.execute("PRAGMA database_list").fetchone()[2])
        ignored_dirs = {".git", "logs", "__pycache__", ".pytest_cache"}
        ignored_names = {db_path.name, "memory.db"}
        data_dirs = {repo / "blood", repo / "datasets"}

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
                if not initial:
                    logging.info("Hash for %s changed", file)
                    if any(d in file.parents for d in data_dirs):
                        data_changed_bytes += file.stat().st_size
                    else:
                        code_changed = True
                self.set_meta(key, digest)

        if code_changed:
            self.set_meta("needs_training", "1")

        if data_changed_bytes:
            total = int(self.get_meta("data_pending_bytes") or "0")
            total += data_changed_bytes
            training_limit = int(
                os.getenv("LE_TRAINING_LIMIT_BYTES", str(5 * 1024))
            )
            if total >= training_limit:
                self.set_meta("needs_training", "1")
                total = 0
            self.set_meta("data_pending_bytes", str(total))

    def needs_training(self) -> bool:
        """Return True if retraining is required."""
        return self.get_meta("needs_training") == "1"

    def get_accumulated_size(self) -> int:
        """Return number of bytes accumulated toward the training limit."""
        return int(self.get_meta("data_pending_bytes") or "0")

    @staticmethod
    def hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
