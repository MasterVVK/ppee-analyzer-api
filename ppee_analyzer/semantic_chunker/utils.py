"""
Вспомогательные функции для семантического разделения документов.
"""

import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple

# Настройка логирования
logger = logging.getLogger(__name__)


def detect_docling_availability() -> bool:
    """
    Проверяет доступность библиотеки docling.

    Returns:
        bool: True если docling доступен, иначе False
    """
    try:
        import docling
        from docling.document_converter import DocumentConverter
        return True
    except ImportError:
        logger.warning("Библиотека docling не установлена. Некоторые функции будут недоступны.")
        return False


def detect_gpu_availability() -> bool:
    """
    Проверяет доступность CUDA для работы с GPU.

    Returns:
        bool: True если GPU доступен, иначе False
    """
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            logger.info(f"CUDA доступна. Используется GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("CUDA недоступна. Используется CPU.")
        return has_cuda
    except ImportError:
        logger.info("PyTorch не установлен. Используется CPU.")
        return False


def is_likely_table_continuation(content: str) -> bool:
    """
    Определяет, является ли текст продолжением таблицы.

    Args:
        content: Текст для анализа

    Returns:
        bool: True если текст похож на продолжение таблицы, иначе False
    """
    # Признаки продолжения таблицы
    table_indicators = [
        r'\d+\.\s*\w+',  # Нумерация (29. Конструкция выпуска)
        r'^\d+',  # Начинается с числа
        r'Координаты:',  # Специфические слова
        r'^\s*[А-Яа-я\s\-]+$',  # Только текст (возможно заголовок колонки)
        r'соответствии',  # Признак продолжения текста
        r'^[А-Я][а-я]+\s+с',  # Начинается с заглавной буквы и предлога
        r'\|\s*$',  # Признак таблицы
    ]

    for indicator in table_indicators:
        if re.search(indicator, content.strip()[:100]):
            return True
    return False


def identify_content_type(text: str) -> str:
    """
    Определяет тип содержимого текста.

    Args:
        text: Текст для анализа

    Returns:
        str: Тип содержимого ('table', 'heading', 'list', 'text')
    """
    if '|' in text and text.count('|') > 4:
        return 'table'
    elif any(line.startswith(('* ', '- ', '+ ')) for line in text.split('\n')):
        return 'list'
    elif text.strip().startswith('##'):
        return 'heading'
    else:
        return 'text'


def extract_section_info(text: str) -> Dict[str, str]:
    """
    Извлекает информацию о разделе и его структуре.

    Args:
        text: Текст для анализа

    Returns:
        Dict[str, str]: Информация о разделе (section, subsection, section_path)
    """
    section = "Не определено"
    subsection = ""
    section_path = ""

    # Поиск номера раздела (например, 4.г)
    section_number_match = re.search(r'(\d+(\.\w+)*)\.\s+', text)
    if section_number_match:
        section_path = section_number_match.group(1)

    # Поиск заголовка раздела
    section_match = re.search(r'##\s+([^\n]+)', text)
    if section_match:
        section = section_match.group(1).strip()

    # Поиск подзаголовка
    subsection_match = re.search(r'###\s+([^\n]+)', text)
    if subsection_match:
        subsection = subsection_match.group(1).strip()

    return {
        "section": section,
        "subsection": subsection,
        "section_path": section_path
    }


def generate_unique_id() -> str:
    """
    Генерирует уникальный идентификатор.

    Returns:
        str: Уникальный идентификатор
    """
    return str(uuid.uuid4())


def decode_unicode_escapes(text: str) -> str:
    """
    Декодирует Unicode escape sequences в тексте.
    Например: /uni041F -> П, /uni044F -> я

    Args:
        text: Текст с Unicode escape sequences

    Returns:
        str: Декодированный текст
    """
    if not text:
        return text

    import re

    def replace_unicode(match):
        # Извлекаем hex код
        hex_code = match.group(1)
        try:
            # Преобразуем в символ
            return chr(int(hex_code, 16))
        except ValueError:
            # Если не удалось, возвращаем как есть
            return match.group(0)

    # Паттерн для поиска /uniXXXX
    pattern = r'/uni([0-9A-Fa-f]{4})'

    # Заменяем все вхождения
    decoded_text = re.sub(pattern, replace_unicode, text)

    return decoded_text


def fix_superscript_units(text: str) -> str:
    """
    Исправляет разбитые надстрочные символы в единицах измерения.
    Например: "м /год\n3" -> "м³/год"

    Args:
        text: Текст с разбитыми надстрочными символами

    Returns:
        str: Исправленный текст
    """
    if not text:
        return text

    import re

    # Сначала обрабатываем случаи, где цифра на следующей строке после единицы измерения
    # Ищем паттерны где есть единица измерения, потом что-то (включая /год/), потом цифра на новой строке

    # Специальный паттерн для случаев типа "м/год/ 365" где 3 на следующей строке
    text = re.sub(
        r'([кмсд]?м)(/[а-яА-Яa-zA-Z]+/?)(\s*[^\n]*)\n\s*3\b',
        r'\1³\2\3',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r'([кмсд]?м)(/[а-яА-Яa-zA-Z]+/?)(\s*[^\n]*)\n\s*2\b',
        r'\1²\2\3',
        text,
        flags=re.MULTILINE
    )

    # Паттерны для распространенных единиц измерения с надстрочными символами
    patterns = [
        # м³, км³, см³ и т.д. с разделителем
        (r'([кмсд]?м)\s*/([а-яА-Я]+)\s*\n\s*3\b', r'\1³/\2'),
        (r'([кмсд]?м)\s*\n\s*3\b', r'\1³'),

        # м², км², см² и т.д. с разделителем
        (r'([кмсд]?м)\s*/([а-яА-Я]+)\s*\n\s*2\b', r'\1²/\2'),
        (r'([кмсд]?м)\s*\n\s*2\b', r'\1²'),

        # Общий паттерн для любых единиц с цифрами 2 или 3
        (r'(\w+)\s*/(\w+)\s*\n\s*([23])\b', r'\1\3/\2'),
        (r'(\w+)\s*\n\s*([23])\b', r'\1\2'),

        # Исправление для случаев типа "10 /с"
        (r'(\d+)\s*/([а-яА-Яa-zA-Z]+)\s*\n\s*([23])\b', r'\1\3/\2'),

        # Удаление лишних пробелов перед слэшем
        (r'\s+/', r'/'),
    ]

    # Применяем все паттерны
    result = text
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result, flags=re.MULTILINE)

    # Заменяем обычные цифры на надстрочные символы где это уместно
    # для единиц измерения (включая случаи без пробела)
    replacements = [
        (r'\b([кмсд]?)м3\b', r'\1м³'),
        (r'\b([кмсд]?)м2\b', r'\1м²'),
        (r'\bО2\b', 'О₂'),  # Для химических формул типа мгО2/л
        (r'\bH2\b', 'H₂'),
        (r'\bCO2\b', 'CO₂'),
        (r'\bSO2\b', 'SO₂'),
        (r'\bNO2\b', 'NO₂'),
        (r'\bN2\b', 'N₂'),
        (r'\bO2\b', 'O₂'),
    ]

    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result)

    # Специальная обработка для БПК5, БПК20 и т.д.
    result = re.sub(r'БПК(\d+)', r'БПК₅', result) if 'БПК5' in result else result

    return result