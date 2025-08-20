import asyncio
import csv
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

from inhale_exhale import inhale, exhale, memory
import metrics

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
# Resolve and ensure the working directory exists and is writable
WORK_DIR = Path(os.getenv("LE_WORK_DIR", "names")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
if not os.access(WORK_DIR, os.W_OK):
    raise PermissionError(f"Cannot write to {WORK_DIR}")
SAMPLE_TIMEOUT = int(os.getenv("LE_SAMPLE_TIMEOUT", "120"))
TRAINING_TASK: asyncio.Task | None = None
TRAINING_LIMIT_BYTES = int(
    os.getenv("LE_TRAINING_LIMIT_BYTES", str(5 * 1024))
)
TOP_K = int(os.getenv("LE_TOP_K", "50"))
TEMPERATURE = float(os.getenv("LE_TEMPERATURE", "1.0"))


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
                "--type",
                "transformer",
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
                "--top-k",
                str(TOP_K),
                "--temperature",
                str(TEMPERATURE),
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
        reply = question
        await update.message.reply_text(reply)
        inhale(question, reply)
        await exhale(update.effective_chat.id, context)
        return
    dataset_path = build_dataset()
    try:
        seed = random.randint(0, 2**31 - 1)
        proc = await asyncio.create_subprocess_exec(
            "python",
            "le.py",
            "--type",
            "transformer",
            "-i",
            str(dataset_path),
            "--work-dir",
            str(WORK_DIR),
            "--sample-only",
            "--prompt",
            question,
            "--num-samples",
            "1",
            "--seed",
            str(seed),
            "--quiet",
            "--top-k",
            str(TOP_K),
            "--temperature",
            str(TEMPERATURE),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SAMPLE_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            logging.exception("Sampling timed out")
            reply = "Sampling timed out."
        else:
            if proc.returncode == 0:
                lines = [
                    line
                    for line in stdout.decode().splitlines()
                    if line.strip()
                ]
                reply = lines[-1] if lines else "No output from LE."
            else:
                logging.error("Sampling failed: %s", stderr.decode())
                reply = f"Error: Process exited with status {proc.returncode}"
    except Exception as exc:
        logging.exception("Sampling error")
        reply = f"Error: {exc}"
    finally:
        dataset_path.unlink(missing_ok=True)
    await update.message.reply_text(reply)
    inhale(question, reply)
    await exhale(update.effective_chat.id, context)


async def run_training(
    chat_id: int | None,
    context: ContextTypes.DEFAULT_TYPE | None,
    extra_dataset: Path | None = None,
) -> None:
    dataset_path = build_dataset()
    if extra_dataset:
        try:
            with dataset_path.open("a", encoding="utf-8") as dst, extra_dataset.open(
                "r", encoding="utf-8"
            ) as src:
                dst.write("\n")
                dst.write(src.read())
        except OSError:
            logging.exception("Failed to append extra dataset")
    try:
        memory.set_meta("needs_training", "0")
        proc = await asyncio.create_subprocess_exec(
            "python",
            "le.py",
            "--type",
            "transformer",
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
            return
        if proc.returncode == 0:
            if context and chat_id is not None:
                await context.bot.send_message(
                    chat_id=chat_id, text="Training completed."
                )
            if stdout:
                logging.debug("Training stdout: %s", stdout.decode())
            if stderr:
                logging.debug("Training stderr: %s", stderr.decode())
            warmup_model()
        else:
            logging.error(
                "Training failed with code %s", proc.returncode
            )
            if stdout:
                logging.error("stdout: %s", stdout.decode())
            if stderr:
                logging.error("stderr: %s", stderr.decode())
    except Exception:
        logging.exception("Training error")
    finally:
        dataset_path.unlink(missing_ok=True)


def fine_tune(extra_dataset: Path) -> None:
    """Launch fine-tuning with an additional dataset."""
    global TRAINING_TASK
    if TRAINING_TASK and not TRAINING_TASK.done():
        logging.info("Training already in progress; skipping fine-tune trigger")
        return
    TRAINING_TASK = asyncio.create_task(run_training(None, None, extra_dataset))


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TRAINING_TASK
    if TRAINING_TASK and not TRAINING_TASK.done():
        await update.message.reply_text("Training already in progress.")
        return
    await update.message.reply_text("Training started...")
    chat_id = update.effective_chat.id
    TRAINING_TASK = asyncio.create_task(run_training(chat_id, context))


def build_dataset(latest_line: str | None = None) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", encoding="utf-8"
    ) as tmp:
        total = 0
        seen: set[str] = set()

        def write_line(line: str) -> None:
            nonlocal total
            remaining = TRAINING_LIMIT_BYTES - total
            if remaining <= 0:
                return
            data_bytes = (line + "\n").encode("utf-8")
            tmp.write(data_bytes[:remaining].decode("utf-8", errors="ignore"))
            total += min(len(data_bytes), remaining)

        for directory in ("blood", "datasets"):
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
            for file in dir_path.rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() in {".txt", ".md"}:
                    write_line(file.read_text(encoding="utf-8"))
                elif file.suffix.lower() == ".csv":
                    with file.open(newline="", encoding="utf-8") as f:
                        reader = csv.reader(f)
                        for row in reader:
                            text_cells: list[str] = []
                            for cell in row:
                                cell = cell.strip()
                                if not cell:
                                    continue
                                try:
                                    float(cell)
                                except ValueError:
                                    text_cells.append(cell)
                            if text_cells:
                                write_line(" ".join(text_cells))

        for line in memory.get_messages():
            if line in seen:
                continue
            write_line(line)
            metrics.log_response_metrics(line, 0)
            seen.add(line)

        if latest_line and latest_line not in seen:
            write_line(latest_line)
            metrics.log_response_metrics(latest_line, 0)

    return Path(tmp.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    try:
        warmup_model()
        app.run_polling()
    finally:
        memory.close()


if __name__ == "__main__":
    main()
