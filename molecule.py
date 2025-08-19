import asyncio
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from inhale_exhale import inhale, exhale

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
WORK_DIR = Path(os.getenv("LE_WORK_DIR", "names"))
SAMPLE_TIMEOUT = int(os.getenv("LE_SAMPLE_TIMEOUT", "120"))
TRAINING_TASK: asyncio.Task | None = None
TRAINING_LIMIT_BYTES = 20 * 1024


def warmup_model() -> None:
    """Run a sample to load the model and warm up caches."""
    model_path = WORK_DIR / "model.pt"
    if not model_path.exists():
        return
    dataset_path = build_dataset()
    try:
        subprocess.run(
            [
                "python",
                "le.py",
                "-i",
                str(dataset_path),
                "--work-dir",
                str(WORK_DIR),
                "--sample-only",
                "--num-samples",
                "1",
                "--seed",
                "0",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=SAMPLE_TIMEOUT,
        )
    except Exception:
        logging.exception("Warmup failed")
    finally:
        dataset_path.unlink(missing_ok=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a message and I'll ask LE to respond."
    )


async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TRAINING_TASK
    question = update.message.text
    model_path = WORK_DIR / "model.pt"
    if not model_path.exists():
        if TRAINING_TASK is None or TRAINING_TASK.done():
            TRAINING_TASK = asyncio.create_task(run_training(None, None))
        dataset_path = build_dataset()
        try:
            data = dataset_path.read_text(encoding="utf-8").splitlines()
            lines = [line.strip() for line in data if line.strip()]
            reply = random.choice(lines) if lines else "No training data."
        finally:
            dataset_path.unlink(missing_ok=True)
        await update.message.reply_text(reply)
        inhale(question, reply)
        await exhale(update.effective_chat.id, context)
        return
    dataset_path = build_dataset()
    try:
        seed = random.randint(0, 2**31 - 1)
        result = subprocess.run(
            [
                "python",
                "le.py",
                "-i",
                str(dataset_path),
                "--work-dir",
                str(WORK_DIR),
                "--sample-only",
                "--num-samples",
                "1",
                "--seed",
                str(seed),
                "--quiet",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=SAMPLE_TIMEOUT,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        reply = lines[-1] if lines else "No output from LE."
    except subprocess.TimeoutExpired:
        logging.exception("Sampling timed out")
        reply = "Sampling timed out."
    except Exception as exc:
        logging.exception("Sampling error")
        reply = f"Error: {exc}"
    finally:
        dataset_path.unlink(missing_ok=True)
    await update.message.reply_text(reply)
    inhale(question, reply)
    await exhale(update.effective_chat.id, context)


async def run_training(
    chat_id: int | None, context: ContextTypes.DEFAULT_TYPE | None
) -> None:
    dataset_path = build_dataset()
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "le.py",
            "-i",
            str(dataset_path),
            "--work-dir",
            str(WORK_DIR),
            "--max-steps",
            "200",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=600
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logging.exception("Training timed out")
            if context and chat_id is not None:
                await context.bot.send_message(
                    chat_id=chat_id, text="Training timed out."
                )
            return
        if proc.returncode == 0:
            if context and chat_id is not None:
                await context.bot.send_message(
                    chat_id=chat_id, text="Training completed."
                )
        else:
            logging.error("Training failed: %s", stderr.decode())
            if context and chat_id is not None:
                await context.bot.send_message(
                    chat_id=chat_id, text="Training failed."
                )
    except Exception:
        logging.exception("Training error")
        if context and chat_id is not None:
            await context.bot.send_message(
                chat_id=chat_id, text="Training error."
            )
    finally:
        dataset_path.unlink(missing_ok=True)


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TRAINING_TASK
    if TRAINING_TASK and not TRAINING_TASK.done():
        await update.message.reply_text("Training already in progress.")
        return
    await update.message.reply_text("Training started...")
    chat_id = update.effective_chat.id
    TRAINING_TASK = asyncio.create_task(run_training(chat_id, context))


def build_dataset() -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", encoding="utf-8"
    ) as tmp:
        total = 0
        for txt_file in Path("blood").glob("*.txt"):
            data_bytes = (
                txt_file.read_text(encoding="utf-8") + "\n"
            ).encode("utf-8")
            remaining = TRAINING_LIMIT_BYTES - total
            if remaining <= 0:
                break
            tmp.write(
                data_bytes[:remaining].decode("utf-8", errors="ignore")
            )
            total += min(len(data_bytes), remaining)
    return Path(tmp.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    warmup_model()
    app.run_polling()


if __name__ == "__main__":
    main()
