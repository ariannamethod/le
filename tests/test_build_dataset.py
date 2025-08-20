import importlib
import tg
import memory


import pytest


@pytest.mark.parametrize("limit", [1024, 2048])
def test_build_dataset_enforces_limit(tmp_path, monkeypatch, limit):
    monkeypatch.setenv("LE_TRAINING_LIMIT_BYTES", str(limit))
    importlib.reload(tg)
    mem = memory.Memory(str(tmp_path / "memory.db"))
    monkeypatch.setattr(tg, "memory", mem)

    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    chunk = "a" * (limit // 2)
    (blood_dir / "f1.txt").write_text(chunk)
    (blood_dir / "f2.txt").write_text(chunk)

    monkeypatch.chdir(tmp_path)
    dataset_path = tg.build_dataset()
    try:
        assert dataset_path.stat().st_size == limit
    finally:
        dataset_path.unlink()
        mem.close()


def test_build_dataset_includes_memory_and_question(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    (blood_dir / "base.txt").write_text("base\n")
    importlib.reload(tg)
    mem = memory.Memory(str(tmp_path / "memory.db"))
    mem.record_message("q1", "a1")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tg, "memory", mem)
    dataset_path = tg.build_dataset("q2")
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        assert lines[-1] == "q2"
        assert "q1" in lines and "a1" in lines
        assert "base" in lines
    finally:
        dataset_path.unlink()
        mem.close()


def test_build_dataset_reads_various_file_types(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    (blood_dir / "b.txt").write_text("blood text")

    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "d.md").write_text("dataset md")
    (datasets_dir / "d.csv").write_text("num,text\n1,hello\n2,world\n")

    importlib.reload(tg)
    mem = memory.Memory(str(tmp_path / "memory.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tg, "memory", mem)
    dataset_path = tg.build_dataset()
    try:
        content = dataset_path.read_text(encoding="utf-8")
        lines = content.splitlines()
        assert "blood text" in lines
        assert "dataset md" in lines
        assert "hello" in lines
        assert "world" in lines
        assert "1" not in content and "2" not in content
    finally:
        dataset_path.unlink()
        mem.close()


def test_build_dataset_reads_nested_files(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    (blood_dir / "b.txt").write_text("blood text")

    nested_dir = tmp_path / "datasets" / "sub"
    nested_dir.mkdir(parents=True)
    (nested_dir / "n.txt").write_text("nested")

    importlib.reload(tg)
    mem = memory.Memory(str(tmp_path / "memory.db"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tg, "memory", mem)
    dataset_path = tg.build_dataset()
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        assert "nested" in lines
    finally:
        dataset_path.unlink()
        mem.close()


def test_build_dataset_deduplicates_memory(tmp_path, monkeypatch):
    importlib.reload(tg)
    mem = memory.Memory(str(tmp_path / "memory.db"))
    mem.record_message("hello", "a1")
    mem.record_message("hello", "a2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tg, "memory", mem)
    dataset_path = tg.build_dataset()
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        assert lines.count("hello") == 1
    finally:
        dataset_path.unlink()
        mem.close()


def test_update_repo_hash_respects_env_limit(tmp_path, monkeypatch):
    limit = 100
    monkeypatch.setenv("LE_TRAINING_LIMIT_BYTES", str(limit))
    monkeypatch.chdir(tmp_path)
    mem = memory.Memory(str(tmp_path / "memory.db"))

    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    (blood_dir / "f1.txt").write_text("a" * (limit // 2))
    mem.update_repo_hash()

    (blood_dir / "f2.txt").write_text("a" * limit)
    mem.update_repo_hash()

    try:
        assert mem.needs_training()
    finally:
        mem.close()
