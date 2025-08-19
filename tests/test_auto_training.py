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

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        return None

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
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
            assert any(
                "file.txt" in record.message for record in caplog.records
            )
        finally:
            os.chdir(cwd)


def test_update_repo_hash_ignores_temp_files(tmp_path, monkeypatch):
    fake_mol = types.ModuleType("molecule")

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        return None

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        inhale_exhale.memory = mem
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            (tmp_path / "logs").mkdir()
            log = tmp_path / "logs" / "app.log"
            mem.update_repo_hash()
            mem.set_meta("needs_training", "0")
            log.write_text("a")
            mem.update_repo_hash()
            assert not mem.needs_training()
            log.write_text("b")
            mem.update_repo_hash()
            assert not mem.needs_training()
        finally:
            os.chdir(cwd)


@pytest.mark.asyncio
async def test_exhale_triggers_training_when_needed(tmp_path, monkeypatch):
    fake_mol = types.ModuleType("molecule")
    event = asyncio.Event()

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        event.set()
        inhale_exhale.memory.set_meta("needs_training", "0")

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        inhale_exhale.memory = mem
        mem.set_meta("needs_training", "1")
        await inhale_exhale.exhale(1, None)
        assert mem.needs_training()
        assert fake_mol.TRAINING_TASK is not None
        await fake_mol.TRAINING_TASK
        assert event.is_set()
        assert not mem.needs_training()


@pytest.mark.asyncio
async def test_exhale_skips_when_not_needed(tmp_path, monkeypatch):
    fake_mol = types.ModuleType("molecule")
    started = {"flag": False}

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        started["flag"] = True

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        inhale_exhale.memory = mem
        mem.set_meta("needs_training", "0")
        await inhale_exhale.exhale(1, None)
        assert not started["flag"]


@pytest.mark.asyncio
async def test_startup_triggers_training_when_model_missing(
    tmp_path, monkeypatch
):
    fake_mol = types.ModuleType("molecule")
    event = asyncio.Event()

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        event.set()

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)
    monkeypatch.chdir(tmp_path)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))

    await asyncio.sleep(0)
    assert fake_mol.TRAINING_TASK is not None
    await fake_mol.TRAINING_TASK
    assert event.is_set()
    inhale_exhale.memory.close()


@pytest.mark.asyncio
async def test_retrains_on_successive_data_additions(tmp_path, monkeypatch):
    fake_mol = types.ModuleType("molecule")
    calls: list[int] = []

    async def dummy_run_training(chat_id, context, extra_dataset=None):
        calls.append(1)
        inhale_exhale.memory.set_meta("needs_training", "0")

    fake_mol.run_training = dummy_run_training
    fake_mol.TRAINING_TASK = None
    monkeypatch.setitem(sys.modules, "molecule", fake_mol)

    inhale_exhale = importlib.reload(importlib.import_module("inhale_exhale"))
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        inhale_exhale.memory = mem
        cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            (tmp_path / "datasets").mkdir()
            mem.update_repo_hash(tmp_path)
            mem.set_meta("needs_training", "0")

            (tmp_path / "datasets" / "a.txt").write_text("a" * (10 * 1024 + 1))
            mem.update_repo_hash(tmp_path)
            await inhale_exhale.exhale(1, None)
            assert fake_mol.TRAINING_TASK is not None
            await fake_mol.TRAINING_TASK

            (tmp_path / "datasets" / "b.txt").write_text("b" * (10 * 1024 + 1))
            mem.update_repo_hash(tmp_path)
            await inhale_exhale.exhale(1, None)
            assert fake_mol.TRAINING_TASK is not None
            await fake_mol.TRAINING_TASK
        finally:
            os.chdir(cwd)

    assert len(calls) == 2
