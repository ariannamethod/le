import os
from pathlib import Path
from typing import Set

LOG_PATH = os.getenv("RESPONSE_LOG_PATH")

def _load_phrases() -> Set[str]:
    if LOG_PATH and Path(LOG_PATH).exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()

def is_unique(phrase: str) -> bool:
    """Return True if ``phrase`` has not been logged yet."""
    return phrase not in _load_phrases()

def log_phrase(phrase: str) -> None:
    """Append ``phrase`` to the log file if logging is enabled."""
    if not LOG_PATH:
        return
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(phrase + "\n")

def check_and_log(phrase: str) -> bool:
    """Check uniqueness and log ``phrase`` if it is new.

    Returns ``True`` when ``phrase`` was not seen before.
    """
    if is_unique(phrase):
        log_phrase(phrase)
        return True
    return False
