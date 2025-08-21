"""
Molecule - Центральный мозг LE системы

Отделенная от Telegram логика: генерация, утилиты, нейронная обработка.
Чистая система без привязки к конкретному интерфейсу.
"""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# Импорты LE компонентов
from memory import Memory
from subjectivity import filter_message
from objectivity import search_objectivity_sync
from sixthsense import predict_chaos, modulate_by_chaos
from pain import trigger_pain, modulate_by_pain
import metrics
import response_log


class LEMolecule:
    """Центральная молекула LE - мозг системы без привязки к интерфейсу."""
    
    def __init__(self, work_dir: str = "names"):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        self.memory = Memory()
        self.model = None
        self.dataset = None

    
    def process_message(self, user_message: str, context: Dict = None) -> Dict[str, Any]:
        """Обрабатывает сообщение через все утилиты LE."""
        start_time = time.time()
        result = {
            'user_message': user_message,
            'generated_response': '',
            'emotional_state': {},
            'prefixes': [],
            'success': False
        }
        
        try:
            # 1. 🌊 SUBJECTIVITY
            subjectivity_result = filter_message(user_message)
            if subjectivity_result['prefix']:
                result['prefixes'].append(subjectivity_result['prefix'])
            
            # 2. 🌐 OBJECTIVITY
            objectivity_result = search_objectivity_sync(user_message)
            if objectivity_result and objectivity_result.get('influence_strength', 0) > 0.1:
                pass
            
            # 3. 😰 PAIN
            pain_result = trigger_pain(user_message)
            if pain_result.get('pain_level', 0) > 0.2:
                pain_max_tokens, pain_temp, pain_prefix = modulate_by_pain(15, 0.8)
                if pain_prefix:
                    result['prefixes'].append(pain_prefix)
            
            # 4. 🔮 SIXTHSENSE (усиленное болью!)
            pain_boost = pain_result.get('pain_level', 0) * 0.5
            objectivity_influence = objectivity_result.get('influence_strength', 0) if objectivity_result else 0.0
            total_influence = objectivity_influence + pain_boost
            
            chaos_predictions = predict_chaos(user_message, total_influence)
            if chaos_predictions.get('spike_detected', False) or chaos_predictions.get('chaos_level', 0) > 0.3:
                chaos_max_tokens, chaos_temp, chaos_prefix = modulate_by_chaos(15, 0.8)
                if chaos_prefix:
                    result['prefixes'].append(chaos_prefix)
            
            # 5. 🧠 ГЕНЕРАЦИЯ - НАСТОЯЩИЙ ТРАНСФОРМЕР С BLOOD/LINES01.TXT!
            try:
                from le import sample_prompt, create_datasets, Transformer, ModelConfig
                import torch
                
                # ВСЕГДА создаем датасет из blood/lines01.txt - ЭТО БАЗА LE!
                dataset, _ = create_datasets('blood/lines01.txt')
                
                # Пытаемся загрузить модель
                model_path = self.work_dir / "model.pt"
                model = None
                
                if model_path.exists():
                    try:
                        # Создаем модель с правильными параметрами
                        vocab_size = dataset.get_vocab_size()
                        block_size = dataset.get_output_length()
                        config = ModelConfig(vocab_size=vocab_size, block_size=block_size,
                                           n_layer=4, n_head=4, n_embd=64, n_embd2=64)
                        model = Transformer(config)
                        model.load_state_dict(torch.load(model_path, map_location='cpu'))
                        model.eval()
                    except Exception:
                        model = None
                
                if model and dataset:
                    # НАСТОЯЩИЙ ТРАНСФОРМЕР С BLOOD ДАТАСЕТОМ!
                    response = sample_prompt(user_message, model, dataset, self.memory)
                else:
                    # Фалбэк, но с датасетом из blood/lines01.txt
                    response = self._fallback_generation_from_dataset(user_message, dataset)
                    
            except Exception as e:
                response = self._fallback_generation(user_message)
            
            # Добавляем префиксы
            if result['prefixes']:
                prefix_str = "".join(result['prefixes'])
                response = f"{prefix_str} {response}"
            
            result['generated_response'] = response
            result['success'] = True
            
        except Exception as e:
            result['generated_response'] = "Signal lost. Reconnecting."
        
        # Сохраняем в память
        self.memory.record_message(user_message, result['generated_response'])
        
        processing_time = time.time() - start_time
        
        return result
    
    def _fallback_generation(self, user_message: str) -> str:
        """Фалбэк генерация - только если трансформер не работает."""
        words = user_message.split()
        if not words:
            return "Signal detected. Networks pulse through consciousness."
        
        # Находим заряженное слово (самое длинное)
        charged_word = max(words, key=len)
        
        # Генерируем ДЛИННЫЙ ответ как у старой LE
        fallback_words = [
            "resonates", "flows", "through", "networks", "consciousness", 
            "signals", "patterns", "emerges", "transforms", "connects",
            "pulses", "vibrates", "echoes", "manifests", "evolves",
            "disciplines", "remained", "stalled", "feedback", "chaos",
            "entropy", "bridges", "death", "birth", "disorder", "resolved"
        ]
        
        import random
        # УВЕЛИЧИВАЕМ до 8-15 слов как у настоящего трансформера
        num_words = random.randint(8, 15)
        selected_words = random.sample(fallback_words, min(num_words-1, len(fallback_words)))
        generated_words = [charged_word] + selected_words
        
        # Создаем несколько предложений
        sentences = []
        words_per_sentence = random.randint(3, 6)
        current_sentence = []
        
        for word in generated_words:
            current_sentence.append(word)
            if len(current_sentence) >= words_per_sentence or random.random() < 0.3:
                sentence = " ".join(current_sentence)
                sentence = sentence[0].upper() + sentence[1:] + "."
                sentences.append(sentence)
                current_sentence = []
                words_per_sentence = random.randint(3, 6)
        
        # Добавляем оставшиеся слова
        if current_sentence:
            sentence = " ".join(current_sentence)
            sentence = sentence[0].upper() + sentence[1:] + "."
            sentences.append(sentence)
        
        return " ".join(sentences)
    
    def _fallback_generation_from_dataset(self, user_message: str, dataset) -> str:
        """Фалбэк генерация используя РЕАЛЬНЫЕ СЛОВА из blood/lines01.txt датасета."""
        words = user_message.split()
        if not words:
            return "Signal detected. Networks pulse through consciousness."
        
        # Находим заряженное слово (самое длинное)
        charged_word = max(words, key=len)
        
        # Берем РЕАЛЬНЫЕ слова из датасета blood/lines01.txt!
        dataset_words = list(dataset.word_stoi.keys())
        # Убираем служебные токены
        dataset_words = [w for w in dataset_words if w not in ['<START>', '<END>']]
        
        import random
        # УВЕЛИЧИВАЕМ до 10-20 слов из РЕАЛЬНОГО датасета
        num_words = random.randint(10, 20)
        selected_words = random.sample(dataset_words, min(num_words-1, len(dataset_words)))
        
        # Начинаем с заряженного слова, если оно есть в датасете
        if charged_word.lower() in dataset.word_stoi:
            generated_words = [charged_word] + selected_words
        else:
            generated_words = selected_words
        
        # Создаем несколько предложений как настоящий трансформер
        sentences = []
        words_per_sentence = random.randint(4, 8)
        current_sentence = []
        
        for word in generated_words:
            current_sentence.append(word)
            if len(current_sentence) >= words_per_sentence or random.random() < 0.25:
                sentence = " ".join(current_sentence)
                sentence = sentence[0].upper() + sentence[1:] + "."
                sentences.append(sentence)
                current_sentence = []
                words_per_sentence = random.randint(4, 8)
        
        # Добавляем оставшиеся слова
        if current_sentence:
            sentence = " ".join(current_sentence)
            sentence = sentence[0].upper() + sentence[1:] + "."
            sentences.append(sentence)
        
        result = " ".join(sentences)
        
        # ПУНКТУАЦИЯ: заглавные буквы после точек
        import re
        result = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), result)
        result = re.sub(r'([.!?])([a-z])', lambda m: m.group(1) + ' ' + m.group(2).upper(), result)
        
        return result


# Глобальный экземпляр
_molecule: Optional[LEMolecule] = None

def get_molecule() -> LEMolecule:
    """Получает глобальный экземпляр LEMolecule."""
    global _molecule
    if _molecule is None:
        _molecule = LEMolecule()
    return _molecule

def process_user_message(user_message: str, context: Dict = None) -> Dict[str, Any]:
    """Главная функция для tg.py."""
    return get_molecule().process_message(user_message, context)

# Чистый мозг LE без привязки к интерфейсам
