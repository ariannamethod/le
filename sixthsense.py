"""
SixthSense - Предсказание хаотических спайков для LE

Биологически вдохновленный модуль для предсказания непредсказуемых моментов
в диалогах и влияния на генерацию LE через динамический хаос.
"""

import time
import random
import math
from typing import Dict, Tuple, Optional
from pathlib import Path


class SixthSense:
    """Модуль предсказания хаотических спайков, вдохновлённый биологической интуицией."""
    
    def __init__(self, chaos_base: float = 0.5, sensitivity: float = 0.2, memory_decay: float = 0.95):
        self.chaos = chaos_base  # Базовый уровень хаоса (0.0-1.0)
        self.sensitivity = sensitivity  # Чувствительность к внешним сигналам
        self.memory_decay = memory_decay  # Скорость затухания памяти
        self.last_update = time.time()
        self.spike_history = []  # История спайков
        self.conversation_pulse = 0.5  # Пульс разговора
        
    def calculate_conversation_pulse(self, user_message: str, previous_responses: list = None) -> float:
        """Вычисляет пульс разговора на основе сообщения."""
        if not user_message:
            return 0.3
            
        # Анализ интенсивности сообщения
        intensity_factors = [
            len(user_message) / 100,  # Длина
            user_message.count('!') * 0.1,  # Восклицания
            user_message.count('?') * 0.08,  # Вопросы
            len([w for w in user_message.split() if w.isupper()]) * 0.05,  # КАПС
        ]
        
        # Эмоциональные маркеры
        emotional_words = ['love', 'hate', 'amazing', 'terrible', 'wow', 'omg', 'блять', 'охуенно']
        emotion_boost = sum(0.1 for word in emotional_words if word.lower() in user_message.lower())
        
        base_pulse = min(sum(intensity_factors) + emotion_boost, 1.0)
        
        # Учитываем историю разговора
        if previous_responses:
            # Если предыдущие ответы были короткими - повышаем пульс
            avg_length = sum(len(r) for r in previous_responses[-3:]) / max(len(previous_responses[-3:]), 1)
            if avg_length < 50:  # Короткие ответы
                base_pulse += 0.2
        
        return max(0.1, min(base_pulse, 1.0))
    
    def detect_chaos_patterns(self, user_message: str) -> float:
        """Детектит паттерны хаоса в сообщении пользователя."""
        chaos_indicators = [
            # Философские темы
            any(word in user_message.lower() for word in 
                ['consciousness', 'reality', 'existence', 'meaning', 'soul', 'mind']),
            
            # Творческие запросы
            any(word in user_message.lower() for word in 
                ['create', 'imagine', 'dream', 'art', 'poetry', 'music']),
            
            # Эмоциональная нестабильность
            len([c for c in user_message if c in '!?']) > 2,
            
            # Противоречия или парадоксы
            'but' in user_message.lower() and 'not' in user_message.lower(),
            
            # Метафизические вопросы
            user_message.lower().startswith(('what if', 'why do', 'how can', 'what is')),
        ]
        
        chaos_level = sum(chaos_indicators) / len(chaos_indicators)
        return chaos_level
    
    def foresee(self, user_message: str, influence: float = 0.0, 
                external_context: Dict = None) -> Dict[str, float]:
        """
        Предсказывает хаотический спайк на основе сообщения и контекста.
        
        Args:
            user_message: Сообщение пользователя
            influence: Внешнее влияние (например, от objectivity)
            external_context: Дополнительный контекст
            
        Returns:
            Dict с предсказаниями и параметрами
        """
        current_time = time.time()
        time_delta = current_time - self.last_update
        
        # Обновляем пульс разговора
        self.conversation_pulse = self.calculate_conversation_pulse(user_message)
        
        # Детектим хаос в сообщении
        message_chaos = self.detect_chaos_patterns(user_message)
        
        # Биологический шум (имитация нейронной активности)
        neural_noise = random.uniform(-0.05, 0.05)
        
        # Временной фактор (усталость/возбуждение системы)
        time_factor = min(time_delta / 3600, 1.0) * 0.1  # Час = максимум
        
        # Интеграция всех факторов
        delta = (
            message_chaos * 0.4 +
            influence * self.sensitivity +
            self.conversation_pulse * 0.2 +
            time_factor +
            neural_noise
        )
        
        # Обновляем уровень хаоса с затуханием памяти
        old_chaos = self.chaos
        self.chaos = max(0.05, min(self.chaos * self.memory_decay + delta, 0.95))
        
        # Детектируем спайк
        spike_detected = abs(self.chaos - old_chaos) > 0.3
        if spike_detected:
            self.spike_history.append({
                'time': current_time,
                'intensity': self.chaos,
                'trigger': 'message_chaos' if message_chaos > 0.5 else 'influence'
            })
            # Ограничиваем историю
            self.spike_history = self.spike_history[-10:]
        
        self.last_update = current_time
        
        # Предсказания для модуляции LE
        predictions = {
            'chaos_level': self.chaos,
            'spike_detected': spike_detected,
            'conversation_pulse': self.conversation_pulse,
            'message_chaos': message_chaos,
            'neural_noise': neural_noise,
            'time_since_last': time_delta
        }
        
        return predictions
    
    def modulate_generation_params(self, base_max_tokens: int = 15, 
                                 base_temperature: float = 0.8) -> Tuple[int, float, str]:
        """
        Модулирует параметры генерации LE на основе предсказанного хаоса.
        
        Returns:
            (max_tokens, temperature, prefix_emoji)
        """
        # Эмоджи для разных состояний
        if self.chaos > 0.7:
            prefix = "⚡️"  # Высокий хаос - вихрь
            multiplier = 1.0 + (self.chaos - 0.7) * 2  # До 1.6x токенов
            temp_boost = (self.chaos - 0.7) * 0.8  # До +0.24 температуры
            
        elif self.chaos > 0.4:
            prefix = ""  # Средний хаос - спайк энергии
            multiplier = 1.0 + (self.chaos - 0.4) * 0.8  # До 1.24x токенов
            temp_boost = (self.chaos - 0.4) * 0.4  # До +0.12 температуры
            
        elif self.chaos > 0.2:
            prefix = ""  # Низкий хаос - интуиция
            multiplier = 1.0
            temp_boost = 0.0
            
        else:
            prefix = ""  # Очень низкий хаос - без эмоджи
            multiplier = 0.8  # Короче
            temp_boost = -0.1  # Консервативнее
        
        max_tokens = int(base_max_tokens * multiplier)
        temperature = max(0.3, min(base_temperature + temp_boost, 1.3))
        
        return max_tokens, temperature, prefix
    
    def get_spike_insights(self) -> Dict:
        """Возвращает инсайты о недавних спайках."""
        if not self.spike_history:
            return {'recent_spikes': 0, 'avg_intensity': 0.0, 'trend': 'stable'}
        
        recent = [s for s in self.spike_history if time.time() - s['time'] < 300]  # 5 минут
        
        if recent:
            avg_intensity = sum(s['intensity'] for s in recent) / len(recent)
            trend = 'rising' if len(recent) >= 2 and recent[-1]['intensity'] > recent[-2]['intensity'] else 'falling'
        else:
            avg_intensity = 0.0
            trend = 'stable'
        
        return {
            'recent_spikes': len(recent),
            'avg_intensity': avg_intensity,
            'trend': trend,
            'total_spikes': len(self.spike_history)
        }
    
    def reset(self):
        """Сброс состояния хаоса к базовому."""
        self.chaos = 0.5
        self.conversation_pulse = 0.5
        self.last_update = time.time()
        self.spike_history.clear()
    
    def get_state(self) -> Dict:
        """Возвращает полное состояние для логирования."""
        return {
            'chaos_level': self.chaos,
            'conversation_pulse': self.conversation_pulse,
            'last_update': self.last_update,
            'spike_count': len(self.spike_history),
            'insights': self.get_spike_insights()
        }


# Глобальный экземпляр
_sixth_sense_instance: Optional[SixthSense] = None


def get_sixth_sense() -> SixthSense:
    """Получает глобальный экземпляр SixthSense (singleton)."""
    global _sixth_sense_instance
    if _sixth_sense_instance is None:
        _sixth_sense_instance = SixthSense()
    return _sixth_sense_instance


def predict_chaos(user_message: str, external_influence: float = 0.0) -> Dict[str, any]:
    """Удобная функция для предсказания хаоса."""
    sense = get_sixth_sense()
    predictions = sense.foresee(user_message, external_influence)
    
    # Логируем интересные моменты
    if predictions['spike_detected']:
        print(f"🌀 Chaos spike detected! Level: {predictions['chaos_level']:.2f}")
    
    return predictions


def modulate_by_chaos(base_max_tokens: int = 15, base_temperature: float = 0.8) -> Tuple[int, float, str]:
    """Модулирует параметры генерации на основе текущего хаоса."""
    sense = get_sixth_sense()
    return sense.modulate_generation_params(base_max_tokens, base_temperature)


# Живая система без шаблонов - только чистое предчувствие хаоса
