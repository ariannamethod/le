import molecule


def test_build_dataset_enforces_limit(tmp_path, monkeypatch):
    blood_dir = tmp_path / "blood"
    blood_dir.mkdir()
    chunk = "a" * (molecule.TRAINING_LIMIT_BYTES // 2)
    (blood_dir / "f1.txt").write_text(chunk)
    (blood_dir / "f2.txt").write_text(chunk)
    monkeypatch.chdir(tmp_path)
    dataset_path = molecule.build_dataset()
    try:
        assert dataset_path.stat().st_size == molecule.TRAINING_LIMIT_BYTES
    finally:
        dataset_path.unlink()
