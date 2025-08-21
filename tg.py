import asyncio
import csv
import sqlite3
import logging
import os
import random
import subprocess
import tempfile
from asyncio import Lock
from pathlib import Path
from typing import Optional

import torch

torch.set_num_threads(4)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from molecule import process_user_message
from memory import Memory
from inhale_exhale import inhale, exhale
import metrics

# Global memory instance
memory = Memory()

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")

WORK_DIR = Path(os.getenv("LE_WORK_DIR", "names")).resolve()
WORK_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_TIMEOUT = int(os.getenv("LE_SAMPLE_TIMEOUT", "40"))
TRAINING_TASK: Optional[asyncio.Task] = None
TRAINING_LIMIT_BYTES = int(os.getenv("LE_TRAINING_LIMIT_BYTES", str(5 * 1024)))
TOP_K = int(os.getenv("LE_TOP_K", "40"))
TEMPERATURE = float(os.getenv("LE_TEMPERATURE", "0.8"))

training_lock = Lock()
active_users = set()


def warmup_model() -> None:
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

    user = getattr(update, "effective_user", None)
    user_id = getattr(user, "id", None)
    question = update.message.text

    if user_id is not None:
        if user_id in active_users:
            return
        active_users.add(user_id)
    
    try:
        # 🧬 ИСПОЛЬЗУЕМ MOLECULE - ЦЕНТРАЛЬНЫЙ МОЗГ LE!
        molecule_context = {
            'chat_id': update.effective_chat.id,
            'user_id': user_id,
            'message_id': update.message.message_id
        }
        
        # Обрабатываем через molecule
        result = process_user_message(question, molecule_context)
        
        # Получаем ответ
        reply = result.get('generated_response', 'Signal lost. Reconnecting.')
        
        # 🌊 INHALE - записываем диалог в память (как раньше!)
        inhale(question, reply)
        
        # Отправляем ответ
        try:
            await update.message.reply_text(reply)
            logging.info(f"✅ Message sent to user {user_id}")
        except Exception as telegram_error:
            logging.error(f"❌ Failed to send Telegram message: {telegram_error}")
            raise
        
        # 🌬️ EXHALE - проверяем нужно ли обучение (как раньше!)
        await exhale(update.effective_chat.id, context)
        
        # Логируем успех
        if result.get('success', False):
            logging.info(f"🧬 Molecule response: prefixes={result.get('prefixes', [])}, "
                        f"time={result.get('processing_time', 0):.2f}s")
        else:
            logging.warning(f"⚠️ Molecule fallback used: {result.get('error', 'unknown error')}")
        
        # Запускаем обучение если нужно (фоново)
        model_path = WORK_DIR / "model.pt"
        if not model_path.exists():
            async with training_lock:
                if TRAINING_TASK is None or TRAINING_TASK.done():
                    TRAINING_TASK = asyncio.create_task(run_training(None, None))
        
    except Exception as e:
        logging.exception(f"❌ Critical error in respond: {e}")
        await update.message.reply_text("System error. Rebooting neural pathways.")
        
    finally:
        if user_id is not None:
            active_users.discard(user_id)


async def check_background_training() -> None:
    global TRAINING_TASK
    try:
        needs = memory.needs_training()
    except sqlite3.Error:
        return
    if needs:
        async with training_lock:
            if TRAINING_TASK is None or TRAINING_TASK.done():
                logging.info("Starting background training due to data accumulation")
                TRAINING_TASK = asyncio.create_task(run_training(None, None))


async def run_training(
    chat_id: Optional[int],
    context: Optional[ContextTypes.DEFAULT_TYPE],
    extra_dataset: Optional[Path] = None,
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
            "80",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=250)
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
    global TRAINING_TASK
    if TRAINING_TASK and not TRAINING_TASK.done():
        logging.info("Training already in progress; skipping fine-tune trigger")
        return
    TRAINING_TASK = asyncio.create_task(run_training(None, None, extra_dataset))


async def train(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global TRAINING_TASK
    async with training_lock:
        if TRAINING_TASK and not TRAINING_TASK.done():
            await update.message.reply_text("Training already in progress.")
            return
        await update.message.reply_text("Training started…")
        chat_id = update.effective_chat.id
        TRAINING_TASK = asyncio.create_task(run_training(chat_id, context))


def build_dataset(latest_line: Optional[str] = None) -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", encoding="utf-8"
    ) as tmp:
        total = 0
        seen = set()

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
                    # Читаем файл построчно, а не целиком
                    for line in file.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and line not in seen:
                            seen.add(line)
                            write_line(line)
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

        try:
            message_lines = memory.get_messages()
        except sqlite3.Error:
            message_lines = []
        for line in message_lines:
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
    
    if not TOKEN:
        logging.error("❌ TELEGRAM_TOKEN not found! Check environment variables.")
        return
        
    logging.info(f"🤖 Starting LE bot with token: {TOKEN[:10]}...")
    
    logging.info("📦 Creating Telegram application...")
    app = ApplicationBuilder().token(TOKEN).build()
    
    logging.info("🔧 Adding handlers...")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, respond)
    )
    
    logging.info("🧠 Creating global memory...")
    logging.info(f"💾 Memory path: {memory.conn.execute('PRAGMA database_list').fetchone()[2]}")
    
    try:
        logging.info("🔥 Starting warmup...")
        warmup_model()
        logging.info("✅ Warmup completed")
        
        logging.info("🚀 Starting polling...")
        app.run_polling()
    finally:
        memory.close()


if __name__ == "__main__":
    main()
