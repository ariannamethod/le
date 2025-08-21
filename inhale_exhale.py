import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path

from memory import Memory

WORK_DIR = Path(os.getenv("LE_WORK_DIR", "names"))
MODEL_PATH = WORK_DIR / "model.pt"

# Global memory instance
memory = Memory()


def startup_training_check() -> None:
    """Schedule training at startup if the model file is missing."""
    if not MODEL_PATH.exists():
        logging.info("Model file missing at startup; training will be triggered on first message")


def _startup_training_check() -> None:
    """Schedule training at startup if the model file is missing.""" 
    startup_training_check()


asyncio.get_event_loop().call_soon(_startup_training_check)


def inhale(question: str, answer: str) -> None:
    """Record the latest conversation and update repository hash."""
    memory.record_message(question, answer)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().isoformat()
    log_file = log_dir / "conversations.txt"
    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(f"{timestamp}\t{question}\t{answer}\n")

    memory.update_repo_hash()


async def exhale(chat_id: int, context) -> None:
    """Trigger training in the background if required."""
    if not memory.needs_training():
        return

    import tg  # Local import to avoid circular dependency

    if tg.TRAINING_TASK is None or tg.TRAINING_TASK.done():
        logging.info("Starting background training")
        tg.TRAINING_TASK = asyncio.create_task(
            tg.run_training(chat_id, context)
        )
    else:
        logging.info("Training already running; will retrigger if still needed")

        def _retry(_task: asyncio.Task) -> None:
            asyncio.create_task(exhale(chat_id, context))

        if not getattr(tg.TRAINING_TASK, "_retry_set", False):
            tg.TRAINING_TASK.add_done_callback(_retry)
            setattr(tg.TRAINING_TASK, "_retry_set", True)
