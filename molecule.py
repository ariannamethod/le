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
        print("🧬 LEMolecule initialized - brain online")
    
    def process_message(self, user_message: str, context: Dict = None) -> Dict[str, Any]:
        """Обрабатывает сообщение через все утилиты LE."""
        start_time = time.time()
        print(f"🧬 Processing: '{user_message[:30]}...'")
        
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
                result['prefixes'].append("🌐")
            
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
            
            # 5. 🧠 ГЕНЕРАЦИЯ
            from le import sample_prompt
            if hasattr(self, 'model') and hasattr(self, 'dataset') and self.model and self.dataset:
                response = sample_prompt(user_message, self.model, self.dataset, self.memory)
            else:
                response = self._fallback_generation(user_message)
            
            # Добавляем префиксы
            if result['prefixes']:
                prefix_str = "".join(result['prefixes'])
                response = f"{prefix_str} {response}"
            
            result['generated_response'] = response
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Molecule error: {e}")
            result['generated_response'] = "Signal lost. Reconnecting."
        
        # Сохраняем в память
        self.memory.record_message(user_message, result['generated_response'])
        
        processing_time = time.time() - start_time
        print(f"🧬 Complete: {processing_time:.2f}s, prefixes: {''.join(result['prefixes'])}")
        
        return result
    
    def _fallback_generation(self, user_message: str) -> str:
        """Простая генерация без модели."""
        words = user_message.split()
        charged_word = max(words, key=len) if words else "mystery"
        
        fallback_words = ["resonates", "through", "networks", "consciousness", "flows"]
        import random
        generated_words = [charged_word] + random.sample(fallback_words, 2)
        
        text = " ".join(generated_words)
        return text[0].upper() + text[1:] + "." if text else "Mystery unfolds."


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
