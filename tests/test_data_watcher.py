import os

from memory import Memory


def test_large_data_file_triggers_flag(tmp_path):
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            data_dir = tmp_path / "datasets"
            data_dir.mkdir()
            big = data_dir / "big.bin"
            big.write_bytes(b"a" * (11 * 1024))
            mem.update_repo_hash()
            assert mem.needs_training()
        finally:
            os.chdir(cwd)


def test_small_files_accumulate_to_trigger(tmp_path):
    with Memory(path=str(tmp_path / "mem.db")) as mem:
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            data_dir = tmp_path / "datasets"
            data_dir.mkdir()
            small1 = data_dir / "one.bin"
            small1.write_bytes(b"a" * (5 * 1024))
            mem.update_repo_hash()
            assert not mem.needs_training()
            small2 = data_dir / "two.bin"
            small2.write_bytes(b"a" * (6 * 1024))
            mem.update_repo_hash()
            assert mem.needs_training()
        finally:
            os.chdir(cwd)

