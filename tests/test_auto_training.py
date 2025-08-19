import asyncio
import logging
import os
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import inhale_exhale
from memory import Memory


def test_update_repo_hash_detects_changes(tmp_path, caplog):
    mem = Memory(path=str(tmp_path / "mem.db"))
    inhale_exhale.memory = mem
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        file = Path("file.txt")
        file.write_text("a")
        mem.update_repo_hash()
        mem.set_meta("needs_training", "0")
        file.write_text("b")
        with caplog.at_level(logging.INFO):
            mem.update_repo_hash()
        assert mem.needs_training()
        assert any("file.txt" in record.message for record in caplog.records)
    finally:
        os.chdir(cwd)


@pytest.mark.asyncio
async def test_exhale_triggers_training_when_needed(tmp_path, monkeypatch):
    mem = Memory(path=str(tmp_path / "mem.db"))
    inhale_exhale.memory = mem
    mem.set_meta("needs_training", "1")

    fake_mol = types.ModuleType("molecule")
    event = asyncio.Event()

    async def dummy_run_training(chat_id, context):
        event.set()

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    await inhale_exhale.exhale(1, None)
    assert fake_mol.TRAINING_TASK is not None
    await fake_mol.TRAINING_TASK
    assert event.is_set()
    assert not mem.needs_training()
