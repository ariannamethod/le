import os
import asyncio
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


MODEL_READY = asyncio.Event()


async def ensure_model() -> None:
    model_path = Path("names/model.pt")
    if model_path.exists():
        MODEL_READY.set()
        return
    proc = await asyncio.create_subprocess_exec(
        "python",
        "le.py",
        "-i",
        "blood/lines01.txt",
        "-o",
        "names",
        "--max-steps",
        "200",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    if proc.returncode == 0 and model_path.exists():
        MODEL_READY.set()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a message and I'll ask LE to respond."
    )


async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await MODEL_READY.wait()
    try:
        proc = await asyncio.create_subprocess_exec(
            "python",
            "le.py",
            "-i",
            "blood/lines01.txt",
            "-o",
            "names",
            "--sample-only",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            reply = f"Error: {stderr.decode().strip()}"
        else:
            lines = [
                line for line in stdout.decode().splitlines()
                if line.strip()
            ]
            reply = (
                "\n".join(lines[-10:]) if lines else "No output from LE."
            )
    except Exception as exc:
        reply = f"Error: {exc}"
    await update.message.reply_text(reply)


async def post_init(app):
    app.create_task(ensure_model())


def main() -> None:
    app = ApplicationBuilder().token(TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    app.run_polling()


if __name__ == "__main__":
    main()
