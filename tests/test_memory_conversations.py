from memory import Memory


def test_get_conversations_returns_context(tmp_path):
    mem = Memory(str(tmp_path / "mem.db"))
    try:
        mem.record_message("q1", "a1", "ctx1")
        mem.record_message("q2", "a2", "ctx2")
        assert mem.get_conversations() == [
            ("ctx1", "q1", "a1"),
            ("ctx2", "q2", "a2"),
        ]
    finally:
        mem.close()
