import subprocess
import types

import pytest

import molecule


async def _noop_run_training(chat_id, context):
    return None


molecule.run_training = _noop_run_training
molecule.TRAINING_TASK = None


@pytest.mark.asyncio
async def test_respond_produces_one_line(monkeypatch, tmp_path):
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    (names_dir / "model.pt").write_text("dummy")
    molecule.WORK_DIR = names_dir
    monkeypatch.chdir(tmp_path)

    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("hello\n")
    monkeypatch.setattr(molecule, "build_dataset", lambda: dataset_file)

    monkeypatch.setattr(molecule, "inhale", lambda q, r: None)

    async def dummy_exhale(chat_id, context):
        return None
    monkeypatch.setattr(molecule, "exhale", dummy_exhale)

    captured = {}

    def fake_run(args, **kwargs):
        captured['args'] = args

        class Res:
            stdout = "noise\nsample\n\n"

        return Res()
    monkeypatch.setattr(molecule.subprocess, "run", fake_run)

    replies = []

    class DummyMessage:
        text = "hi"

        async def reply_text(self, text):
            replies.append(text)

    update = types.SimpleNamespace(
        message=DummyMessage(), effective_chat=types.SimpleNamespace(id=1)
    )
    await molecule.respond(update, None)

    assert replies == ["sample"]
    assert "--num-samples" in captured['args'] and "1" in captured['args']
    assert "--quiet" in captured['args']
    assert "--work-dir" in captured['args'] and str(names_dir) in captured['args']
    assert "\n" not in replies[0]


@pytest.mark.asyncio
async def test_respond_handles_timeout(monkeypatch, tmp_path):
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    (names_dir / "model.pt").write_text("dummy")
    molecule.WORK_DIR = names_dir
    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("hello\n")
    monkeypatch.setattr(molecule, "build_dataset", lambda: dataset_file)
    monkeypatch.setattr(molecule, "inhale", lambda q, r: None)

    async def dummy_exhale(chat_id, context):
        return None

    monkeypatch.setattr(molecule, "exhale", dummy_exhale)

    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr(molecule.subprocess, "run", fake_run)

    replies = []

    class DummyMessage:
        text = "hi"

        async def reply_text(self, text):
            replies.append(text)

    update = types.SimpleNamespace(
        message=DummyMessage(), effective_chat=types.SimpleNamespace(id=1)
    )
    await molecule.respond(update, None)

    assert replies == ["Sampling timed out."]
