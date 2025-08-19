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


def _startup_training_check() -> None:
    """Schedule training at startup if the model file is missing."""
    if not MODEL_PATH.exists():
        import molecule  # Local import to avoid circular dependency

        if molecule.TRAINING_TASK is None or molecule.TRAINING_TASK.done():
            logging.info("Model file missing; starting initial training")
            molecule.TRAINING_TASK = asyncio.create_task(
                molecule.run_training(None, None)
            )


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

    import molecule  # Local import to avoid circular dependency

    if molecule.TRAINING_TASK is None or molecule.TRAINING_TASK.done():
        logging.info("Starting background training")
        molecule.TRAINING_TASK = asyncio.create_task(
            molecule.run_training(chat_id, context)
        )
    else:
        logging.info("Training already running; will retrigger if still needed")

        def _retry(_task: asyncio.Task) -> None:
            asyncio.create_task(exhale(chat_id, context))

        if not getattr(molecule.TRAINING_TASK, "_retry_set", False):
            molecule.TRAINING_TASK.add_done_callback(_retry)
            setattr(molecule.TRAINING_TASK, "_retry_set", True)
