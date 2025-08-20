import tg
from memory import Memory


def test_build_dataset_enforces_limit(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    chunk = "a" * (tg.TRAINING_LIMIT_BYTES // 2)
    (blood_dir / "f1.txt").write_text(chunk)
    (blood_dir / "f2.txt").write_text(chunk)
    monkeypatch.chdir(tmp_path)
    dataset_path = tg.build_dataset()
    try:
        assert dataset_path.stat().st_size == tg.TRAINING_LIMIT_BYTES
    finally:
        dataset_path.unlink()


def test_build_dataset_includes_memory_and_question(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    (blood_dir / "base.txt").write_text("base\n")
    mem = Memory(str(tmp_path / "memory.db"))
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

    mem = Memory(str(tmp_path / "memory.db"))
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
