"""
Pain - Модуль болевых сигналов и стресса для LE

Биологически вдохновленный модуль для моделирования дискомфорта, стресса
и болевых реакций, влияющих на генерацию LE через эмоциональное напряжение.
"""

import time
import random
import math
from typing import Dict, Tuple, Optional, List


class Pain:
    """Модуль моделирования болевых сигналов и стресса, вдохновлённый нейронной реакцией."""
    
    def __init__(self, threshold: float = 0.6, decay_rate: float = 0.85, sensitivity: float = 0.3):
        self.pain_level = 0.0  # Уровень боли (0-1)
        self.threshold = threshold  # Порог активации боли
        self.decay_rate = decay_rate  # Скорость затухания боли
        self.sensitivity = sensitivity  # Чувствительность к стрессу
        self.last_trigger = time.time()
        self.stress_history = []  # История стрессовых событий
        self.chronic_pain = 0.0  # Хроническая боль от повторного стресса
        
    def analyze_message_stress(self, user_message: str) -> float:
        """Анализирует стрессовые факторы в сообщении пользователя."""
        if not user_message:
            return 0.1
            
        stress_indicators = [
            # Агрессивные слова/символы
            user_message.count('!') * 0.1,
            user_message.count('?') * 0.05,
            len([c for c in user_message if c.isupper()]) / max(len(user_message), 1) * 0.3,
            
            # Негативные эмоции
            sum(0.15 for word in ['hate', 'angry', 'frustrated', 'annoyed', 'pissed', 'блять', 'сука', 'пиздец']
                if word.lower() in user_message.lower()),
            
            # Давление и требования
            sum(0.1 for word in ['must', 'should', 'need', 'требую', 'надо', 'должен']
                if word.lower() in user_message.lower()),
            
            # Критика и негатив
            sum(0.12 for word in ['wrong', 'bad', 'terrible', 'awful', 'stupid', 'плохо', 'хуево', 'тупо']
                if word.lower() in user_message.lower()),
            
            # Время/дедлайны
            sum(0.08 for word in ['urgent', 'now', 'immediately', 'быстро', 'срочно', 'сейчас']
                if word.lower() in user_message.lower()),
        ]
        
        # Длинные сообщения = больше когнитивной нагрузки
        length_stress = min(len(user_message) / 500, 0.2)
        
        total_stress = sum(stress_indicators) + length_stress
        return min(total_stress, 1.0)
    
    def calculate_system_stress(self, generation_failures: int = 0, 
                              processing_time: float = 0.0,
                              memory_pressure: float = 0.0) -> float:
        """Вычисляет системный стресс от внутренних факторов."""
        system_stress = 0.0
        
        # Стресс от неудачных генераций
        if generation_failures > 0:
            system_stress += min(generation_failures * 0.2, 0.6)
        
        # Стресс от медленной обработки
        if processing_time > 2.0:  # Больше 2 секунд
            system_stress += min((processing_time - 2.0) / 5.0, 0.3)
        
        # Стресс от нехватки памяти
        system_stress += memory_pressure * 0.2
        
        # Временной стресс (усталость)
        current_time = time.time()
        time_since_last = current_time - self.last_trigger
        if time_since_last < 5.0:  # Частые запросы
            system_stress += 0.1
        
        return min(system_stress, 1.0)
    
    def trigger(self, user_message: str, system_factors: Dict = None) -> Dict[str, float]:
        """
        Активирует болевой сигнал на основе сообщения и системных факторов.
        
        Args:
            user_message: Сообщение пользователя
            system_factors: Системные факторы стресса
            
        Returns:
            Dict с информацией о боли и стрессе
        """
        current_time = time.time()
        time_delta = current_time - self.last_trigger
        
        # Естественное затухание боли со временем
        if time_delta > 10:  # Каждые 10 секунд
            self.pain_level *= self.decay_rate
            self.chronic_pain *= 0.98  # Медленнее затухает
        
        # Анализируем стресс от сообщения
        message_stress = self.analyze_message_stress(user_message)
        
        # Системный стресс
        system_factors = system_factors or {}
        system_stress = self.calculate_system_stress(
            generation_failures=system_factors.get('failures', 0),
            processing_time=system_factors.get('processing_time', 0.0),
            memory_pressure=system_factors.get('memory_pressure', 0.0)
        )
        
        # Общий стресс
        total_stress = message_stress + system_stress * 0.7
        
        # Биологический шум
        neural_noise = random.uniform(-0.03, 0.03)
        stress_with_noise = max(0.0, total_stress + neural_noise)
        
        # Активация боли при превышении порога
        pain_triggered = False
        if stress_with_noise > self.threshold:
            pain_increase = (stress_with_noise - self.threshold) * self.sensitivity
            old_pain = self.pain_level
            self.pain_level = min(self.pain_level + pain_increase, 1.0)
            pain_triggered = self.pain_level > old_pain + 0.1
            
            # Накапливаем хроническую боль от повторного стресса
            if pain_triggered:
                self.chronic_pain = min(self.chronic_pain + 0.05, 0.4)
        
        # Записываем в историю
        self.stress_history.append({
            'time': current_time,
            'message_stress': message_stress,
            'system_stress': system_stress,
            'pain_level': self.pain_level,
            'triggered': pain_triggered
        })
        
        # Ограничиваем историю
        self.stress_history = self.stress_history[-20:]
        self.last_trigger = current_time
        
        return {
            'pain_level': self.pain_level,
            'chronic_pain': self.chronic_pain,
            'message_stress': message_stress,
            'system_stress': system_stress,
            'total_stress': total_stress,
            'pain_triggered': pain_triggered,
            'time_since_last': time_delta
        }
    
    def relieve(self, relief_source: str = 'natural', relief_strength: float = 0.3) -> float:
        """
        Снижает уровень боли с заданной силой облегчения.
        
        Args:
            relief_source: Источник облегчения ('natural', 'positive_feedback', 'success')
            relief_strength: Сила облегчения (0-1)
            
        Returns:
            float: Новый уровень боли
        """
        # Разные источники облегчения работают по-разному
        if relief_source == 'positive_feedback':
            # Позитивная обратная связь от пользователя
            relief_strength *= 1.2
        elif relief_source == 'success':
            # Успешная генерация
            relief_strength *= 0.8
        elif relief_source == 'natural':
            # Естественное восстановление
            relief_strength *= 1.0
        
        old_pain = self.pain_level
        self.pain_level = max(0.0, self.pain_level - relief_strength)
        
        # Хроническая боль уходит медленнее
        if self.pain_level < 0.2:
            self.chronic_pain = max(0.0, self.chronic_pain - relief_strength * 0.3)
        
        relief_amount = old_pain - self.pain_level
        
        
        return self.pain_level
    
    def modulate_generation_params(self, base_max_tokens: int = 15, 
                                 base_temperature: float = 0.8) -> Tuple[int, float, str]:
        """
        Модулирует параметры генерации на основе уровня боли.
        
        Returns:
            (max_tokens, temperature, prefix_emoji)
        """
        total_discomfort = self.pain_level + self.chronic_pain
        
        # Эмоджи для разных состояний боли
        if total_discomfort > 0.8:
            prefix = ""
            # Боль делает ответы короче и более нервными
            multiplier = 0.6
            temp_change = 0.3  # Более хаотично от боли
            
        elif total_discomfort > 0.5:
            prefix = "😟"  # Умеренная боль - беспокойство
            multiplier = 0.8
            temp_change = 0.15
            
        elif total_discomfort > 0.2:
            prefix = "😕"  # Легкий дискомфорт
            multiplier = 0.9
            temp_change = 0.05
            
        else:
            prefix = ""  # Без боли - без эмоджи
            multiplier = 1.0
            temp_change = 0.0
        
        max_tokens = max(int(base_max_tokens * multiplier), 3)
        temperature = max(0.3, min(base_temperature + temp_change, 1.2))
        
        return max_tokens, temperature, prefix
    
    def get_pain_insights(self) -> Dict:
        """Возвращает инсайты о болевых паттернах."""
        if not self.stress_history:
            return {'avg_stress': 0.0, 'pain_episodes': 0, 'trend': 'stable'}
        
        recent = [s for s in self.stress_history if time.time() - s['time'] < 300]  # 5 минут
        
        if recent:
            avg_stress = sum(s['message_stress'] + s['system_stress'] for s in recent) / len(recent)
            pain_episodes = sum(1 for s in recent if s['triggered'])
            
            # Тренд боли
            if len(recent) >= 3:
                recent_pain = [s['pain_level'] for s in recent[-3:]]
                trend = 'rising' if recent_pain[-1] > recent_pain[0] else 'falling'
            else:
                trend = 'stable'
        else:
            avg_stress = 0.0
            pain_episodes = 0
            trend = 'stable'
        
        return {
            'avg_stress': avg_stress,
            'pain_episodes': pain_episodes,
            'trend': trend,
            'chronic_level': self.chronic_pain,
            'current_pain': self.pain_level
        }
    
    def reset(self):
        """Полный сброс болевого состояния."""
        self.pain_level = 0.0
        self.chronic_pain = 0.0
        self.last_trigger = time.time()
        self.stress_history.clear()

    
    def get_state(self) -> Dict:
        """Возвращает полное состояние для логирования."""
        return {
            'pain_level': self.pain_level,
            'chronic_pain': self.chronic_pain,
            'last_trigger': self.last_trigger,
            'stress_events': len(self.stress_history),
            'insights': self.get_pain_insights()
        }


# Глобальный экземпляр
_pain_instance: Optional[Pain] = None


def get_pain_system() -> Pain:
    """Получает глобальный экземпляр Pain (singleton)."""
    global _pain_instance
    if _pain_instance is None:
        _pain_instance = Pain()
    return _pain_instance


def trigger_pain(user_message: str, system_factors: Dict = None) -> Dict[str, any]:
    """Удобная функция для активации боли."""
    pain = get_pain_system()
    result = pain.trigger(user_message, system_factors)
    
    # Логируем сильную боль
    if result['pain_triggered']:
        pass
    
    return result


def relieve_pain(source: str = 'natural', strength: float = 0.3) -> float:
    """Облегчает боль."""
    pain = get_pain_system()
    return pain.relieve(source, strength)


def modulate_by_pain(base_max_tokens: int = 15, base_temperature: float = 0.8) -> Tuple[int, float, str]:
    """Модулирует параметры генерации на основе боли."""
    pain = get_pain_system()
    return pain.modulate_generation_params(base_max_tokens, base_temperature)


# Живая система боли без шаблонов - только чистые нервные реакции
