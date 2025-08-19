import importlib
from datetime import datetime
from pathlib import Path

from memory import Memory


def test_inhale_writes_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        inhale_exhale.memory = mem
        inhale_exhale.inhale("question", "answer")
        log_file = Path("logs") / "conversations.txt"
        assert log_file.exists()
        line = log_file.read_text(encoding="utf-8").strip()
        parts = line.split("\t")
        assert parts[1:] == ["question", "answer"]
        datetime.fromisoformat(parts[0])
