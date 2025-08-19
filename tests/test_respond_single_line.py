import asyncio
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
    monkeypatch.setattr(molecule, "build_dataset", lambda q=None: dataset_file)

    monkeypatch.setattr(molecule, "inhale", lambda q, r, c: None)

    async def dummy_exhale(chat_id, context):
        return None
    monkeypatch.setattr(molecule, "exhale", dummy_exhale)

    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = args

        class Proc:
            returncode = 0

            async def communicate(self):
                return (b"noise\nsample\n\n", b"")

            def kill(self):
                pass

        return Proc()

    monkeypatch.setattr(molecule.asyncio, "create_subprocess_exec", fake_exec)

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
    assert "--num-samples" in captured["args"] and "1" in captured["args"]
    assert "--quiet" in captured["args"]
    assert "--work-dir" in captured["args"]
    assert str(names_dir) in captured["args"]
    assert "\n" not in replies[0]


@pytest.mark.asyncio
async def test_respond_handles_timeout(monkeypatch, tmp_path):
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    (names_dir / "model.pt").write_text("dummy")
    molecule.WORK_DIR = names_dir
    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("hello\n")
    monkeypatch.setattr(molecule, "build_dataset", lambda q=None: dataset_file)
    monkeypatch.setattr(molecule, "inhale", lambda q, r, c: None)

    async def dummy_exhale(chat_id, context):
        return None

    monkeypatch.setattr(molecule, "exhale", dummy_exhale)

    async def fake_exec(*args, **kwargs):
        class Proc:
            returncode = 0

            async def communicate(self):
                return (b"", b"")

            def kill(self):
                pass

        return Proc()

    async def fake_wait_for(coro, timeout):
        coro.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(molecule.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(molecule.asyncio, "wait_for", fake_wait_for)

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


@pytest.mark.asyncio
async def test_respond_returns_line_when_model_missing(monkeypatch, tmp_path):
    names_dir = tmp_path / "names"
    names_dir.mkdir()
    molecule.WORK_DIR = names_dir
    monkeypatch.chdir(tmp_path)

    dataset_file = tmp_path / "dataset.txt"
    dataset_file.write_text("line1\nline2\n")
    monkeypatch.setattr(molecule, "build_dataset", lambda q=None: dataset_file)

    started = {"flag": False}

    async def dummy_run_training(chat_id, context):
        started["flag"] = True

    molecule.run_training = dummy_run_training
    molecule.TRAINING_TASK = None

    monkeypatch.setattr(molecule.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(molecule, "inhale", lambda q, r, c: None)

    async def dummy_exhale(chat_id, context):
        return None

    monkeypatch.setattr(molecule, "exhale", dummy_exhale)

    replies = []

    class DummyMessage:
        text = "hi"

        async def reply_text(self, text):
            replies.append(text)

    update = types.SimpleNamespace(
        message=DummyMessage(), effective_chat=types.SimpleNamespace(id=1)
    )

    await molecule.respond(update, None)
    assert replies == ["line1"]
    assert molecule.TRAINING_TASK is not None
    await molecule.TRAINING_TASK
    assert started["flag"]
    assert "\n" not in replies[0]
