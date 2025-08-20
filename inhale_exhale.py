import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from memory import Memory

WORK_DIR = Path(os.getenv(“LE_WORK_DIR”, “names”))
MODEL_PATH = WORK_DIR / “model.pt”
TRAINING_LIMIT_BYTES = int(os.getenv(“LE_TRAINING_LIMIT_BYTES”, str(5 * 1024)))

# Global memory instance

memory = Memory()

def startup_training_check() -> None:
“”“Schedule training at startup if the model file is missing.”””
if not MODEL_PATH.exists():
logging.info(“Model file missing at startup; training will be triggered on first message”)

def inhale(question: str, answer: str) -> None:
“”“Record the latest conversation and update repository hash.”””
memory.record_message(question, answer)

```
# Логирование диалога
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
timestamp = datetime.utcnow().isoformat()
log_file = log_dir / "conversations.txt"
with log_file.open("a", encoding="utf-8") as fh:
    fh.write(f"{timestamp}\t{question}\t{answer}\n")

# Проверяем, накопилось ли достаточно данных для дообучения
current_size = memory.get_accumulated_size()
if current_size >= TRAINING_LIMIT_BYTES:
    memory.set_meta("needs_training", "1")
    logging.info(f"Data accumulated: {current_size} bytes. Training needed.")

memory.update_repo_hash()
```

# НЕ запускаем автоматически при импорте!

# Вместо этого вызов будет в main() если нужно

# startup_training_check()