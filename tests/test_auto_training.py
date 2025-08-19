import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import asyncio  # noqa: E402
import importlib  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402

from memory import Memory  # noqa: E402


def test_update_repo_hash_detects_changes(tmp_path, caplog, monkeypatch):
    fake_mol = types.ModuleType("molecule")

    async def dummy_run_training(chat_id, context):
        return None

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
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
    fake_mol = types.ModuleType("molecule")
    event = asyncio.Event()

    async def dummy_run_training(chat_id, context):
        event.set()

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    mem = Memory(path=str(tmp_path / "mem.db"))
    inhale_exhale.memory = mem
    mem.set_meta("needs_training", "1")

    await inhale_exhale.exhale(1, None)
    assert fake_mol.TRAINING_TASK is not None
    await fake_mol.TRAINING_TASK
    assert event.is_set()
    assert not mem.needs_training()


@pytest.mark.asyncio
async def test_startup_triggers_training_when_model_missing(
    tmp_path, monkeypatch
):
    fake_mol = types.ModuleType("molecule")
    event = asyncio.Event()

    async def dummy_run_training(chat_id, context):
        event.set()

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)
    monkeypatch.chdir(tmp_path)

    importlib.reload(importlib.import_module("inhale_exhale"))

    await asyncio.sleep(0)
    assert fake_mol.TRAINING_TASK is not None
    await fake_mol.TRAINING_TASK
    assert event.is_set()
