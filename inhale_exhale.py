import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from memory import Memory

WORK_DIR = Path(os.getenv("LE_WORK_DIR", "names"))
MODEL_PATH = WORK_DIR / "model.pt"
TRAINING_LIMIT_BYTES = int(os.getenv("LE_TRAINING_LIMIT_BYTES", str(5 * 1024)))

# Global memory instance
memory = Memory()


def startup_training_check() -> None:
    """Schedule initial training if the model file is missing."""
    if MODEL_PATH.exists():
        return

    import tg
    lock = getattr(tg, "training_lock", asyncio.Lock())

    async def _schedule() -> None:
        async with lock:
            task = getattr(tg, "TRAINING_TASK", None)
            if task is None or task.done():
                tg.TRAINING_TASK = asyncio.create_task(tg.run_training(None, None))

    asyncio.get_event_loop().create_task(_schedule())


def inhale(question: str, answer: str) -> None:
    """Record the latest conversation and update repository hash."""
    memory.record_message(question, answer)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    log_file = log_dir / "conversations.txt"
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"{timestamp}\t{question}\t{answer}\n")

    size = len((question + answer).encode("utf-8"))
    total = memory.get_accumulated_size() + size
    if total >= TRAINING_LIMIT_BYTES:
        memory.set_meta("needs_training", "1")
        total = 0
    memory.set_meta("data_pending_bytes", str(total))

    memory.update_repo_hash()


async def exhale(chat_id: int, context) -> None:
    """Trigger training if flagged by memory."""
    if not memory.needs_training():
        return

    import tg
    lock = getattr(tg, "training_lock", asyncio.Lock())

    async with lock:
        task = getattr(tg, "TRAINING_TASK", None)
        if task is None or task.done():
            tg.TRAINING_TASK = asyncio.create_task(
                tg.run_training(chat_id, context)
            )


startup_training_check()

