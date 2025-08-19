import logging
import os
import subprocess
import sys
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


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a message and I'll ask LE to respond."
    )


async def respond(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        model_path = Path("names/model.pt")
        if not model_path.exists():
            subprocess.run(
                [
                    "python",
                    "le.py",
                    "-i",
                    "blood/lines01.txt",
                    "-o",
                    "names",
                    "--max-steps",
                    "200",
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=600,
            )
        result = subprocess.run(
            [
                "python",
                "le.py",
                "-i",
                "blood/lines01.txt",
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
            line for line in result.stdout.splitlines()
            if line.strip()
        ]
        reply = (
            "\n".join(lines[-10:]) if lines else "No output from LE."
        )
    except Exception as exc:
        reply = f"Error: {exc}"
    await update.message.reply_text(reply)


def main() -> None:
    if not TOKEN:
        logging.error(
            "TELEGRAM_TOKEN is not set. Please define it in a .env file."
        )
        sys.exit(1)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    app.run_polling()


if __name__ == "__main__":
    main()
