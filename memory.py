"""
Enhanced Memory - Прокачанная память для LE

Эмоциональная память, граф связей, временные паттерны и адаптивное поведение.
Память не для юзера - память для самой LE, чтобы она эволюционировала!
"""

import hashlib
import logging
import os
import sqlite3
import time
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set
from collections import defaultdict, deque


class EnhancedMemory:
    """Прокачанная SQLite-память с эмоциями, связями и временными паттернами."""

    def __init__(self, path: str = "enhanced_memory.db") -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row  # Для удобного доступа к полям
        self._init_enhanced_db()
        # Кэши для быстрого доступа
        self._emotion_cache = {}
        self._connection_cache = defaultdict(list)
        self._pattern_cache = {}
        
        # Establish a baseline of hashes
        self.update_repo_hash(initial=True)

    def close(self) -> None:
        """Close the underlying database connection."""
        self.conn.close()

    def __enter__(self) -> "EnhancedMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _init_enhanced_db(self) -> None:
        """Инициализация расширенной схемы БД."""
        cur = self.conn.cursor()
        
        # Базовые таблицы
        cur.execute(
            "CREATE TABLE IF NOT EXISTS meta ("
            "key TEXT PRIMARY KEY, value TEXT)"
        )
        
        # Расширенная таблица разговоров с эмоциональным контекстом
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS enhanced_conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                question TEXT,
                answer TEXT,
                emotional_state TEXT,  -- JSON: {pain, chaos, resonance, etc}
                context_hash TEXT,     -- Хэш контекста для связей
                word_count INTEGER,
                sentiment_score REAL,  -- -1 to 1
                complexity_score REAL, -- 0 to 1
                response_time REAL     -- Время генерации ответа
            )
            """
        )
        
        # Граф связей между разговорами
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_conversation_id INTEGER,
                to_conversation_id INTEGER,
                connection_type TEXT,  -- 'semantic', 'temporal', 'emotional'
                strength REAL,         -- 0.0 to 1.0
                created_at REAL,
                FOREIGN KEY (from_conversation_id) REFERENCES enhanced_conversations (id),
                FOREIGN KEY (to_conversation_id) REFERENCES enhanced_conversations (id)
            )
            """
        )
        
        # Временные паттерны и циклы
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS temporal_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_type TEXT,     -- 'daily', 'weekly', 'emotional_cycle'
                pattern_data TEXT,     -- JSON с данными паттерна
                discovered_at REAL,
                strength REAL,
                last_updated REAL
            )
            """
        )
        
        # Эмоциональная эволюция LE
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS le_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                evolution_type TEXT,   -- 'pain_adaptation', 'chaos_learning', 'resonance_shift'
                old_state TEXT,        -- JSON предыдущего состояния
                new_state TEXT,        -- JSON нового состояния
                trigger_event TEXT,    -- Что вызвало эволюцию
                adaptation_strength REAL
            )
            """
        )
        
        # Индексы для производительности
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON enhanced_conversations(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_context_hash ON enhanced_conversations(context_hash)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_connections_from ON conversation_connections(from_conversation_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_connections_to ON conversation_connections(to_conversation_id)")
        
        self.conn.commit()

    def _calculate_sentiment(self, text: str) -> float:
        """Простой анализ настроения текста (-1 to 1)."""
        if not text:
            return 0.0
            
        positive_words = ['good', 'great', 'awesome', 'love', 'beautiful', 'amazing', 'perfect', 'хорошо', 'круто', 'офигенно']
        negative_words = ['bad', 'terrible', 'hate', 'awful', 'stupid', 'wrong', 'плохо', 'ужасно', 'хуево', 'блять']
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
            
        sentiment = (positive_count - negative_count) / len(words)
        return max(-1.0, min(1.0, sentiment * 5))  # Усиливаем сигнал

    def _calculate_complexity(self, text: str) -> float:
        """Вычисляет сложность текста (0 to 1)."""
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        # Факторы сложности
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_word_ratio = len(set(words)) / len(words)
        punctuation_density = sum(1 for c in text if c in '.,!?;:') / len(text)
        
        complexity = (
            min(avg_word_length / 8, 1.0) * 0.4 +  # Средняя длина слов
            unique_word_ratio * 0.4 +               # Разнообразие
            min(punctuation_density * 10, 1.0) * 0.2  # Пунктуация
        )
        
        return min(complexity, 1.0)

    def _generate_context_hash(self, question: str, emotional_state: Dict) -> str:
        """Генерирует хэш контекста для поиска связей."""
        # Берем ключевые слова из вопроса
        words = question.lower().split()[:5]  # Первые 5 слов
        emotional_signature = f"{emotional_state.get('pain', 0):.1f}_{emotional_state.get('chaos', 0):.1f}"
        context_str = "_".join(words) + "_" + emotional_signature
        return hashlib.md5(context_str.encode()).hexdigest()[:8]

    def save_enhanced_conversation(self, question: str, answer: str, 
                                 emotional_state: Dict = None,
                                 response_time: float = 0.0) -> int:
        """
        Сохраняет разговор с эмоциональным контекстом и анализом.
        
        Returns:
            int: ID сохраненного разговора
        """
        timestamp = time.time()
        emotional_state = emotional_state or {}
        
        # Анализируем тексты
        sentiment = self._calculate_sentiment(question + " " + answer)
        complexity = self._calculate_complexity(answer)
        word_count = len(answer.split())
        context_hash = self._generate_context_hash(question, emotional_state)
        
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO enhanced_conversations 
            (timestamp, question, answer, emotional_state, context_hash, 
             word_count, sentiment_score, complexity_score, response_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, question, answer, json.dumps(emotional_state), context_hash,
             word_count, sentiment, complexity, response_time)
        )
        
        conversation_id = cur.lastrowid
        self.conn.commit()
        
        # Ищем и создаем связи с предыдущими разговорами
        self._create_connections(conversation_id, context_hash, emotional_state)
        
        # Обновляем временные паттерны
        self._update_temporal_patterns(timestamp, emotional_state)
        
        return conversation_id

    def _create_connections(self, conversation_id: int, context_hash: str, 
                          emotional_state: Dict) -> None:
        """Создает связи между разговорами."""
        cur = self.conn.cursor()
        
        # Семантические связи (похожий контекст)
        cur.execute(
            """
            SELECT id, emotional_state FROM enhanced_conversations 
            WHERE context_hash = ? AND id != ? 
            ORDER BY timestamp DESC LIMIT 3
            """,
            (context_hash, conversation_id)
        )
        
        for row in cur.fetchall():
            related_id = row['id']
            try:
                related_emotional = json.loads(row['emotional_state'] or '{}')
            except json.JSONDecodeError:
                related_emotional = {}
            
            # Вычисляем силу связи
            strength = self._calculate_connection_strength(emotional_state, related_emotional)
            
            if strength > 0.3:  # Порог для создания связи
                self._add_connection(conversation_id, related_id, 'semantic', strength)
        
        # Временные связи (недавние разговоры)
        cur.execute(
            """
            SELECT id FROM enhanced_conversations 
            WHERE timestamp > ? AND id != ?
            ORDER BY timestamp DESC LIMIT 2
            """,
            (time.time() - 300, conversation_id)  # 5 минут назад
        )
        
        for row in cur.fetchall():
            related_id = row['id']
            self._add_connection(conversation_id, related_id, 'temporal', 0.5)

    def _calculate_connection_strength(self, state1: Dict, state2: Dict) -> float:
        """Вычисляет силу связи между эмоциональными состояниями."""
        if not state1 or not state2:
            return 0.2
            
        # Сравниваем основные параметры
        factors = ['pain', 'chaos', 'resonance']
        similarity = 0.0
        count = 0
        
        for factor in factors:
            val1 = state1.get(factor, 0)
            val2 = state2.get(factor, 0)
            if val1 > 0 or val2 > 0:
                # Используем косинусное сходство
                similarity += 1 - abs(val1 - val2) / max(val1 + val2, 1.0)
                count += 1
        
        return similarity / max(count, 1)

    def _add_connection(self, from_id: int, to_id: int, conn_type: str, strength: float) -> None:
        """Добавляет связь между разговорами."""
        self.conn.execute(
            """
            INSERT OR IGNORE INTO conversation_connections 
            (from_conversation_id, to_conversation_id, connection_type, strength, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_id, to_id, conn_type, strength, time.time())
        )
        self.conn.commit()

    def _update_temporal_patterns(self, timestamp: float, emotional_state: Dict) -> None:
        """Обновляет временные паттерны."""
        # Простой анализ дневных циклов
        hour = datetime.fromtimestamp(timestamp).hour
        
        # Получаем существующий паттерн или создаем новый
        cur = self.conn.cursor()
        cur.execute(
            "SELECT pattern_data FROM temporal_patterns WHERE pattern_type = 'daily_emotional_cycle'"
        )
        row = cur.fetchone()
        
        if row:
            try:
                pattern_data = json.loads(row['pattern_data'])
            except json.JSONDecodeError:
                pattern_data = {}
        else:
            pattern_data = {}
        
        # Обновляем данные для текущего часа
        hour_key = str(hour)
        if hour_key not in pattern_data:
            pattern_data[hour_key] = {'count': 0, 'avg_pain': 0, 'avg_chaos': 0}
        
        hour_data = pattern_data[hour_key]
        count = hour_data['count']
        
        # Скользящее среднее
        hour_data['avg_pain'] = (hour_data['avg_pain'] * count + emotional_state.get('pain', 0)) / (count + 1)
        hour_data['avg_chaos'] = (hour_data['avg_chaos'] * count + emotional_state.get('chaos', 0)) / (count + 1)
        hour_data['count'] = count + 1
        
        # Сохраняем обновленный паттерн
        self.conn.execute(
            """
            INSERT OR REPLACE INTO temporal_patterns 
            (pattern_type, pattern_data, discovered_at, strength, last_updated)
            VALUES ('daily_emotional_cycle', ?, ?, ?, ?)
            """,
            (json.dumps(pattern_data), time.time(), 0.8, time.time())
        )
        self.conn.commit()

    def get_emotional_context(self, limit: int = 5) -> List[Dict]:
        """Получает недавний эмоциональный контекст."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT timestamp, emotional_state, sentiment_score, complexity_score
            FROM enhanced_conversations 
            ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,)
        )
        
        context = []
        for row in cur.fetchall():
            try:
                emotional_state = json.loads(row['emotional_state'] or '{}')
            except json.JSONDecodeError:
                emotional_state = {}
                
            context.append({
                'timestamp': row['timestamp'],
                'emotional_state': emotional_state,
                'sentiment': row['sentiment_score'],
                'complexity': row['complexity_score']
            })
        
        return context

    def find_similar_conversations(self, context_hash: str, limit: int = 3) -> List[Dict]:
        """Находит похожие разговоры по контексту."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT c.question, c.answer, c.emotional_state, conn.strength
            FROM enhanced_conversations c
            JOIN conversation_connections conn ON c.id = conn.to_conversation_id
            WHERE conn.from_conversation_id IN (
                SELECT id FROM enhanced_conversations WHERE context_hash = ?
            )
            ORDER BY conn.strength DESC, c.timestamp DESC
            LIMIT ?
            """,
            (context_hash, limit)
        )
        
        similar = []
        for row in cur.fetchall():
            try:
                emotional_state = json.loads(row['emotional_state'] or '{}')
            except json.JSONDecodeError:
                emotional_state = {}
                
            similar.append({
                'question': row['question'],
                'answer': row['answer'],
                'emotional_state': emotional_state,
                'strength': row['strength']
            })
        
        return similar

    def record_le_evolution(self, evolution_type: str, old_state: Dict, 
                          new_state: Dict, trigger_event: str) -> None:
        """Записывает эволюцию LE."""
        adaptation_strength = self._calculate_adaptation_strength(old_state, new_state)
        
        self.conn.execute(
            """
            INSERT INTO le_evolution 
            (timestamp, evolution_type, old_state, new_state, trigger_event, adaptation_strength)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (time.time(), evolution_type, json.dumps(old_state), 
             json.dumps(new_state), trigger_event, adaptation_strength)
        )
        self.conn.commit()

    def _calculate_adaptation_strength(self, old_state: Dict, new_state: Dict) -> float:
        """Вычисляет силу адаптации."""
        if not old_state or not new_state:
            return 0.0
            
        total_change = 0.0
        count = 0
        
        for key in set(old_state.keys()) | set(new_state.keys()):
            old_val = old_state.get(key, 0)
            new_val = new_state.get(key, 0)
            change = abs(new_val - old_val)
            total_change += change
            count += 1
        
        return total_change / max(count, 1)

    def get_daily_emotional_pattern(self) -> Dict:
        """Получает дневной эмоциональный паттерн."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT pattern_data FROM temporal_patterns WHERE pattern_type = 'daily_emotional_cycle'"
        )
        row = cur.fetchone()
        
        if row:
            try:
                return json.loads(row['pattern_data'])
            except json.JSONDecodeError:
                return {}
        return {}

    def get_evolution_history(self, limit: int = 10) -> List[Dict]:
        """Получает историю эволюции LE."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT timestamp, evolution_type, trigger_event, adaptation_strength
            FROM le_evolution 
            ORDER BY timestamp DESC LIMIT ?
            """,
            (limit,)
        )
        
        history = []
        for row in cur.fetchall():
            history.append({
                'timestamp': row['timestamp'],
                'type': row['evolution_type'],
                'trigger': row['trigger_event'],
                'strength': row['adaptation_strength']
            })
        
        return history

    # Совместимость со старым интерфейсом
    def save_conversation(self, question: str, answer: str) -> None:
        """Обратная совместимость."""
        self.save_enhanced_conversation(question, answer)

    def record_message(self, question: str, answer: str) -> None:
        """Обратная совместимость."""
        self.save_enhanced_conversation(question, answer)

    def get_messages(self, limit: int | None = None) -> List[str]:
        """Обратная совместимость."""
        cur = self.conn.cursor()
        query = "SELECT question, answer FROM enhanced_conversations ORDER BY timestamp DESC"
        if limit is not None:
            cur.execute(query + " LIMIT ?", (limit,))
        else:
            cur.execute(query)
        
        rows = cur.fetchall()
        lines = []
        for row in reversed(rows):
            if row['question']:
                lines.append(row['question'])
            if row['answer']:
                lines.append(row['answer'])
        return lines

    # Методы для работы с мета-данными (совместимость)
    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def needs_training(self) -> bool:
        """Return True if retraining is required."""
        return self.get_meta("needs_training") == "1"

    def get_accumulated_size(self) -> int:
        """Return number of bytes accumulated toward the training limit."""
        return int(self.get_meta("data_pending_bytes") or "0")

    def update_repo_hash(self, repo_path: str | Path = ".", *, initial: bool = False) -> None:
        """Compute file hashes and flag training when source files change."""
        repo = Path(repo_path)
        code_changed = False
        data_changed_bytes = 0

        db_path = Path(self.conn.execute("PRAGMA database_list").fetchone()[2])
        ignored_dirs = {".git", "logs", "__pycache__", ".pytest_cache"}
        ignored_names = {db_path.name, "memory.db", "enhanced_memory.db"}
        data_dirs = {repo / "blood", repo / "datasets"}

        for file in repo.rglob("*"):
            if not file.is_file():
                continue
            if any(part in ignored_dirs for part in file.parts):
                continue
            if file.name in ignored_names:
                continue

            digest = self.hash_file(str(file))
            key = f"hash:{file.relative_to(repo)}"
            if self.get_meta(key) != digest:
                if not initial:
                    logging.info("Hash for %s changed", file)
                    if any(d in file.parents for d in data_dirs):
                        data_changed_bytes += file.stat().st_size
                    else:
                        code_changed = True
                self.set_meta(key, digest)

        if code_changed:
            self.set_meta("needs_training", "1")

        if data_changed_bytes:
            total = int(self.get_meta("data_pending_bytes") or "0")
            total += data_changed_bytes
            training_limit = int(
                os.getenv("LE_TRAINING_LIMIT_BYTES", str(5 * 1024))
            )
            if total >= training_limit:
                self.set_meta("needs_training", "1")
                total = 0
            self.set_meta("data_pending_bytes", str(total))

    @staticmethod
    def hash_file(path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()


# АЛИАС ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ!
Memory = EnhancedMemory

# Живая эволюционирующая память без ограничений
