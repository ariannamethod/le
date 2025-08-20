import os
from pathlib import Path
from typing import Set

LOG_PATH = os.getenv("RESPONSE_LOG_PATH")

_phrases_cache: Set[str] | None = None


def _load_phrases() -> Set[str]:
    global _phrases_cache
    if _phrases_cache is not None:
        return _phrases_cache

    if LOG_PATH and Path(LOG_PATH).exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8") as f:
                _phrases_cache = {line.strip() for line in f if line.strip()}
        except OSError:
            _phrases_cache = set()
    else:
        _phrases_cache = set()
    return _phrases_cache


def is_unique(phrase: str) -> bool:
    return phrase not in _load_phrases()


def log_phrase(phrase: str) -> None:
    global _phrases_cache
    if not LOG_PATH:
        return
    path = Path(LOG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(phrase + "\n")
        if _phrases_cache is not None:
            _phrases_cache.add(phrase)
    except OSError:
        pass


def check_and_log(phrase: str) -> bool:
    if is_unique(phrase):
        log_phrase(phrase)
        return True
    return False


def clear_cache() -> None:
    global _phrases_cache
    _phrases_cache = None

