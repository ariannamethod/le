import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
TRAINING_TASK: asyncio.Task | None = None


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a message and I'll ask LE to respond."
    )


async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    model_path = Path("names/model.pt")
    if not model_path.exists():
        if TRAINING_TASK and not TRAINING_TASK.done():
            await update.message.reply_text("Training in progress. Please wait.")
        else:
            await update.message.reply_text(
                "Model not trained yet. Send /train to start training."
            )
        return
    dataset_path = build_dataset()
    try:
        result = subprocess.run(
            [
                "python",
                "le.py",
                "-i",
                str(dataset_path),
                "-o",
                "names",
                "--sample-only",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        lines = [
            line for line in result.stdout.splitlines() if line.strip()
        ]
        reply = "\n".join(lines[-10:]) if lines else "No output from LE."
    except Exception as exc:
        logging.exception("Sampling error")
        reply = f"Error: {exc}"
    finally:
        dataset_path.unlink(missing_ok=True)
    await update.message.reply_text(reply)


async def run_training(chat_id: int, context: ContextTypes.DEFAULT_TYPE) -> None:
    dataset_path = build_dataset()
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "le.py",
            "-i",
            str(dataset_path),
            "-o",
            "names",
            "--max-steps",
            "200",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logging.exception("Training timed out")
            await context.bot.send_message(
                chat_id=chat_id, text="Training timed out."
            )
            return
        if proc.returncode == 0:
            await context.bot.send_message(
                chat_id=chat_id, text="Training completed."
            )
        else:
            logging.error("Training failed: %s", stderr.decode())
            await context.bot.send_message(
                chat_id=chat_id, text="Training failed."
            )
    except Exception:
        logging.exception("Training error")
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
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp:
        for txt_file in Path("blood").glob("*.txt"):
            tmp.write(txt_file.read_text())
            tmp.write("\n")
    return Path(tmp.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    app.run_polling()


if __name__ == "__main__":
    main()
