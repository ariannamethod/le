import asyncio
import logging

from memory import Memory

# Global memory instance
memory = Memory()

def inhale(question: str, answer: str) -> None:
    """Record the latest conversation and update repository hash."""
    memory.record_message(question, answer)
    memory.update_repo_hash()


async def exhale(chat_id: int, context) -> None:
    """Trigger training in the background if needed."""
    if memory.needs_training():
        import molecule  # Local import to avoid circular dependency
        if molecule.TRAINING_TASK is None or molecule.TRAINING_TASK.done():
            logging.info("Starting background training")
            molecule.TRAINING_TASK = asyncio.create_task(
                molecule.run_training(chat_id, context)
            )
            memory.set_meta("needs_training", "0")

