"""
Класс для семантического разделения документов ППЭЭ с использованием docling.
"""

import os
import re
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Импорты из текущего модуля
from .models import SemanticChunk, DocumentAnalysisResult
from .utils import (
    detect_docling_availability,
    detect_gpu_availability,
    is_likely_table_continuation,
    identify_content_type,
    extract_section_info,
    generate_unique_id,
    decode_unicode_escapes,  # Импорт функции декодирования
    fix_superscript_units   # Импорт функции исправления надстрочных символов
)

# Импорты для интеграции с ppee_analyzer
from langchain_core.documents import Document

# Настройка логирования
logger = logging.getLogger(__name__)


class SemanticChunker:
    """Класс для семантического разделения документов с использованием docling"""

    def __init__(self, use_gpu: bool = None, threads: int = 24, ocr_languages: List[str] = None):
        """
        Инициализирует чанкер для семантического разделения документов.

        Args:
            use_gpu: Использовать ли GPU (None - автоопределение)
            threads: Количество потоков
            ocr_languages: Список языков для OCR (по умолчанию ["ru", "en"])
        """
        # Устанавливаем OMP_NUM_THREADS перед импортом docling
        import os
        os.environ["OMP_NUM_THREADS"] = str(threads)
        logger.info(f"Установлено OMP_NUM_THREADS={threads}")

        # Проверяем доступность docling
        self.docling_available = detect_docling_availability()
        if not self.docling_available:
            raise ImportError("Библиотека docling не установлена. Установите её для работы с SemanticChunker.")

        self.use_gpu = use_gpu
        self.threads = threads
        self.ocr_languages = ocr_languages or ["ru", "en"]  # По умолчанию русский и английский
        self._converter = None  # Ленивая инициализация
        self._converter_initialized = False

        logger.info(f"SemanticChunker инициализирован (ленивая загрузка), OCR языки: {self.ocr_languages}")

    @property
    def converter(self):
        """Ленивая загрузка конвертера"""
        if not self._converter_initialized:
            logger.info("Инициализация Docling конвертера при первом использовании...")

            # Импортируем docling только при первом использовании
            import docling
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                AcceleratorDevice,
                AcceleratorOptions,
                TableFormerMode,
                EasyOcrOptions
            )

            # Проверяем доступность GPU
            if self.use_gpu is None:
                self.use_gpu = detect_gpu_availability()

            # Настраиваем опции ускорителя
            accelerator_options = AcceleratorOptions(
                num_threads=self.threads,
                device=AcceleratorDevice.CUDA if self.use_gpu else AcceleratorDevice.CPU
            )

            # Настраиваем опции обработки PDF
            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options

            # КРИТИЧНО ДЛЯ ППЭЭ: Включаем обработку таблиц
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            logger.info("Режим таблиц установлен: ACCURATE (важно для технических таблиц ППЭЭ)")

            # Настройка OCR для русского и английского языков
            ocr_options = EasyOcrOptions()
            ocr_options.lang = self.ocr_languages
            ocr_options.force_full_page_ocr = False  # OCR только для изображений
            ocr_options.bitmap_area_threshold = 0.03  # 3% - для мелких схем и таблиц в ППЭЭ
            ocr_options.confidence_threshold = 0.5    # Стандартный порог
            ocr_options.recog_network = 'standard'   # Стандартная сеть распознавания

            pipeline_options.ocr_options = ocr_options
            logger.info(f"OCR настроен для ППЭЭ: языки {self.ocr_languages}, порог площади 3%")
            logger.info(f"OCR будет использовать устройство: {accelerator_options.device}")

            # GPU оптимизации
            if self.use_gpu:
                pipeline_options.accelerator_options.cuda_use_flash_attention2 = True
                logger.info("Flash Attention 2 включен для GPU")

            # НАСТРОЙКИ СПЕЦИАЛЬНО ДЛЯ ППЭЭ:

            # Обязательно включаем
            pipeline_options.do_ocr = True  # OCR для сканированных частей
            pipeline_options.do_table_structure = True  # Структура таблиц критична

            # Включаем для ППЭЭ
            pipeline_options.do_formula_enrichment = True  # ВАЖНО: химические формулы, расчеты
            logger.info("Распознавание формул ВКЛЮЧЕНО (для химических формул в ППЭЭ)")

            # Отключаем ненужное для ППЭЭ
            pipeline_options.do_code_enrichment = False  # В ППЭЭ нет кода
            pipeline_options.do_picture_classification = False  # Не критично
            pipeline_options.do_picture_description = False  # Слишком медленно

            # Отключаем генерацию изображений для скорости
            pipeline_options.generate_page_images = False
            pipeline_options.generate_picture_images = False
            pipeline_options.generate_table_images = False

            # Дополнительные настройки
            pipeline_options.force_backend_text = False

            # Настраиваем конвертер Docling
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options
                    )
                }
            )

            self._converter_initialized = True
            logger.info(f"Docling конвертер инициализирован для ППЭЭ документов")
            logger.info(f"Настройки: GPU={self.use_gpu}, потоки={self.threads}, "
                       f"OCR={self.ocr_languages}, формулы=ВКЛ, таблицы=ACCURATE")

        return self._converter

    def _process_text_content(self, text: str) -> str:
        """
        Обрабатывает текстовое содержимое: декодирует Unicode и исправляет надстрочные символы.

        Args:
            text: Исходный текст

        Returns:
            str: Обработанный текст
        """
        if not text:
            return text

        # Сначала декодируем Unicode escapes
        text = decode_unicode_escapes(text)

        # Затем исправляем надстрочные символы
        #text = fix_superscript_units(text)

        return text

    def extract_chunks(self, pdf_path: str) -> List[Dict]:
        """
        Извлекает и структурирует документ по смысловым блокам.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            List[Dict]: Список смысловых блоков документа
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Файл не найден: {pdf_path}")

        logger.info(f"Начало обработки документа: {pdf_path}")

        # Проверяем OMP_NUM_THREADS
        omp_threads = os.environ.get("OMP_NUM_THREADS", "не установлено")
        logger.info(f"OMP_NUM_THREADS = {omp_threads}")

        # Конвертируем PDF с помощью Docling
        result = self.converter.convert(pdf_path)
        document = result.document

        chunks = []
        current_chunk = {
            "content": "",
            "type": None,
            "page": None,
            "heading": None,
            "table_id": None,
            "all_pages": set()  # Используем set для автоматического удаления дубликатов
        }

        current_table = None
        last_caption = None

        # Словарь для отслеживания статистики по страницам
        pages_encountered = set()

        # Проходим по элементам документа
        for i, (element, level) in enumerate(document.iterate_items()):
            # Определяем страницу
            current_page = None
            if hasattr(element, 'prov') and element.prov and len(element.prov) > 0:
                current_page = element.prov[0].page_no
                pages_encountered.add(current_page)

            # Проверяем, есть ли у элемента атрибут label
            if not hasattr(element, 'label'):
                if hasattr(element, 'text') and element.text.strip():
                    # Обрабатываем текст
                    decoded_text = self._process_text_content(element.text)

                    if current_chunk["content"]:
                        # Преобразуем set в sorted list перед добавлением
                        chunk_to_add = current_chunk.copy()
                        if isinstance(chunk_to_add.get("all_pages"), set):
                            chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                        chunks.append(chunk_to_add)

                    current_chunk = {
                        "content": decoded_text,
                        "type": "unknown",
                        "page": current_page,
                        "heading": None,
                        "table_id": None,
                        "all_pages": {current_page} if current_page else set()
                    }
                continue

            # Определяем тип элемента
            if element.label == "caption" or (
                    element.label == "text" and hasattr(element, 'text') and
                    re.match(r'^Таблица\s*\d+[.:]', element.text, re.IGNORECASE)):
                # Это заголовок таблицы
                if current_chunk["content"] and current_chunk["type"] != "table":
                    chunk_to_add = current_chunk.copy()
                    if isinstance(chunk_to_add.get("all_pages"), set):
                        chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                    chunks.append(chunk_to_add)

                last_caption = self._process_text_content(element.text) if hasattr(element, 'text') else str(element)
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": current_page,
                    "heading": None,
                    "table_id": None,
                    "all_pages": set()
                }

            elif element.label == "table":
                # Обработка таблиц
                table_id = element.self_ref if hasattr(element, 'self_ref') else str(uuid.uuid4())

                # Получаем контент таблицы
                table_content = ""
                try:
                    table_content = element.export_to_markdown(doc=document)
                    table_content = self._process_text_content(table_content)
                except:
                    try:
                        df = element.export_to_dataframe()
                        table_content = df.to_string()
                        table_content = self._process_text_content(table_content)
                    except:
                        table_content = str(element.data) if hasattr(element, 'data') else str(element)
                        table_content = self._process_text_content(table_content)

                # Если есть caption, добавляем его
                if hasattr(element, 'caption_text'):
                    try:
                        caption = element.caption_text(document)
                        if caption and not last_caption:
                            last_caption = self._process_text_content(caption)
                    except:
                        pass

                # Всегда создаем новый чанк для таблицы
                if current_chunk["content"]:
                    chunk_to_add = current_chunk.copy()
                    if isinstance(chunk_to_add.get("all_pages"), set):
                        chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                    chunks.append(chunk_to_add)

                # Создаем чанк для таблицы
                table_chunk = {
                    "content": table_content,
                    "type": "table",
                    "page": current_page,
                    "heading": last_caption,
                    "table_id": table_id,
                    "pages": [current_page] if current_page else [],
                    "all_pages": sorted([current_page]) if current_page else []
                }

                chunks.append(table_chunk)

                # Сбрасываем current_chunk
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": None,
                    "heading": None,
                    "table_id": None,
                    "all_pages": set()
                }
                current_table = None
                last_caption = None

            elif element.label == "heading" or element.label == "section_header":
                # Если это заголовок раздела, начинаем новый чанк
                if current_chunk["content"]:
                    chunk_to_add = current_chunk.copy()
                    if isinstance(chunk_to_add.get("all_pages"), set):
                        chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                    chunks.append(chunk_to_add)

                current_table = None
                last_caption = None

                # Начинаем новый чанк с заголовком
                heading_text = element.text if hasattr(element, 'text') else str(element)
                heading_text = self._process_text_content(heading_text)

                current_chunk = {
                    "content": heading_text,
                    "type": "heading",
                    "page": current_page,
                    "heading": heading_text,
                    "level": level,
                    "table_id": None,
                    "all_pages": {current_page} if current_page else set()
                }

                logger.debug(f"Начинаем новый заголовок: '{heading_text[:50]}...' на странице {current_page}")

            elif element.label == "document_index":
                # Обработка оглавления
                if current_chunk["content"]:
                    chunk_to_add = current_chunk.copy()
                    if isinstance(chunk_to_add.get("all_pages"), set):
                        chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                    chunks.append(chunk_to_add)

                content = ""
                if hasattr(element, 'text'):
                    content = self._process_text_content(element.text)
                elif hasattr(element, 'export_to_markdown'):
                    try:
                        content = element.export_to_markdown(doc=document)
                        content = self._process_text_content(content)
                    except:
                        content = str(element)
                else:
                    content = str(element)

                index_chunk = {
                    "content": content,
                    "type": "document_index",
                    "page": current_page,
                    "heading": "Оглавление",
                    "table_id": None,
                    "all_pages": sorted([current_page]) if current_page else []
                }
                chunks.append(index_chunk)

                # Сбрасываем current_chunk
                current_chunk = {
                    "content": "",
                    "type": None,
                    "page": None,
                    "heading": None,
                    "table_id": None,
                    "all_pages": set()
                }

            elif element.label == "text" or element.label == "paragraph" or element.label == "list-item":
                # Проверяем, не является ли текст подписью к таблице
                if hasattr(element, 'text') and re.match(r'^Таблица\s*\d+[.:]\s*', element.text, re.IGNORECASE):
                    if current_chunk["content"]:
                        chunk_to_add = current_chunk.copy()
                        if isinstance(chunk_to_add.get("all_pages"), set):
                            chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                        chunks.append(chunk_to_add)

                    last_caption = self._process_text_content(element.text)
                    continue

                # Обычный текст или параграф
                current_table = None
                text_content = element.text if hasattr(element, 'text') else str(element)
                text_content = self._process_text_content(text_content)

                if current_chunk["type"] == "heading":
                    # Если предыдущий элемент был заголовком, преобразуем в секцию
                    logger.debug(f"Преобразуем заголовок в секцию, добавляем текст со страницы {current_page}")

                    current_chunk["content"] += "\n\n" + text_content
                    current_chunk["type"] = "section"

                    # Добавляем страницу в set
                    if current_page:
                        current_chunk["all_pages"].add(current_page)
                        logger.debug(
                            f"Секция '{current_chunk['heading'][:30]}...' теперь содержит страницы: {current_chunk['all_pages']}")

                elif current_chunk["type"] == "section":
                    # Продолжаем добавлять к существующей секции
                    logger.debug(
                        f"Добавляем к секции '{current_chunk['heading'][:30]}...' текст со страницы {current_page}")

                    current_chunk["content"] += "\n\n" + text_content

                    # Добавляем страницу в set
                    if current_page:
                        current_chunk["all_pages"].add(current_page)
                        logger.debug(f"Секция теперь содержит страницы: {current_chunk['all_pages']}")

                else:
                    # Начинаем новый текстовый блок
                    if current_chunk["content"]:
                        chunk_to_add = current_chunk.copy()
                        if isinstance(chunk_to_add.get("all_pages"), set):
                            chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                        chunks.append(chunk_to_add)

                    current_chunk = {
                        "content": text_content,
                        "type": "paragraph" if element.label == "paragraph" else element.label,
                        "page": current_page,
                        "heading": None,
                        "table_id": None,
                        "all_pages": {current_page} if current_page else set()
                    }

            else:
                # Для всех остальных типов элементов
                if hasattr(element, 'text') and element.text.strip():
                    decoded_text = self._process_text_content(element.text)

                    if current_chunk["content"]:
                        chunk_to_add = current_chunk.copy()
                        if isinstance(chunk_to_add.get("all_pages"), set):
                            chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
                        chunks.append(chunk_to_add)

                    current_chunk = {
                        "content": decoded_text,
                        "type": element.label,
                        "page": current_page,
                        "heading": None,
                        "table_id": None,
                        "all_pages": {current_page} if current_page else set()
                    }

        # Добавляем последний чанк
        if current_chunk["content"]:
            chunk_to_add = current_chunk.copy()
            if isinstance(chunk_to_add.get("all_pages"), set):
                chunk_to_add["all_pages"] = sorted(list(chunk_to_add["all_pages"]))
            chunks.append(chunk_to_add)

        # Финальная обработка: преобразуем все set в list
        for chunk in chunks:
            if isinstance(chunk.get("all_pages"), set):
                chunk["all_pages"] = sorted(list(chunk["all_pages"]))

            # Добавляем all_pages для чанков, у которых его нет
            if "all_pages" not in chunk:
                if chunk.get("pages"):
                    chunk["all_pages"] = chunk["pages"] if isinstance(chunk["pages"], list) else [chunk["pages"]]
                elif chunk.get("page"):
                    chunk["all_pages"] = [chunk["page"]]
                else:
                    chunk["all_pages"] = []

        # Логирование для отладки
        for i, chunk in enumerate(chunks):
            if chunk.get("type") == "section":
                logger.info(f"Секция {i}: '{chunk.get('heading', 'Без заголовка')[:50]}...' "
                            f"содержит страницы: {chunk.get('all_pages', [])}, "
                            f"основная страница: {chunk.get('page')}")

        logger.info(f"Документ разделен на {len(chunks)} смысловых блоков")
        logger.info(f"Обработано страниц: {sorted(list(pages_encountered))}")

        return chunks

    def post_process_tables(self, chunks: List[Dict]) -> List[Dict]:
        """
        Постобработка таблиц для объединения разорванных на страницах.

        Args:
            chunks: Список чанков документа

        Returns:
            List[Dict]: Обработанные чанки с объединенными таблицами
        """
        processed_chunks = []
        current_table = None

        # Шаг 1: Создание отображения страниц и анализ нумерации
        page_elements = {}
        element_numbers = {}

        for i, chunk in enumerate(chunks):
            page = chunk.get("page")

            if page is not None:
                if page not in page_elements:
                    page_elements[page] = []
                page_elements[page].append(i)

                # Извлекаем любые числовые последовательности, похожие на нумерацию
                import re
                content = chunk.get("content", "")
                # Ищем паттерны нумерации (число с точкой или число с точкой и подпунктом)
                standard_numbers = re.findall(r'\b(\d+)\.\s', content)
                hierarchy_numbers = re.findall(r'\b(\d+)\.(\d+)\.?\s', content)

                if standard_numbers:
                    element_numbers[i] = [int(n) for n in standard_numbers]

        # Шаг 2: Обработка чанков
        for i, chunk in enumerate(chunks):
            chunk_type = chunk.get("type", "")

            # Обработка явных таблиц
            if chunk_type == "table":
                # Определяем, является ли это продолжением предыдущей таблицы
                is_continuation = False

                if current_table is not None:
                    # Проверка 1: Последовательные страницы
                    prev_pages = current_table.get("pages", [current_table.get("page")])
                    if not isinstance(prev_pages, list):
                        prev_pages = [prev_pages] if prev_pages else []

                    curr_page = chunk.get("page")

                    if prev_pages and curr_page:
                        max_prev_page = max(prev_pages)
                        if curr_page == max_prev_page + 1 or curr_page == max_prev_page:
                            # Таблицы с одинаковым или отсутствующим заголовком вероятно связаны
                            if not chunk.get("heading") or chunk.get("heading") == current_table.get("heading"):
                                is_continuation = True

                    # Проверка 2: Структурное сходство таблиц
                    if not is_continuation:
                        # Анализируем структуру таблиц
                        curr_content = chunk.get("content", "")
                        prev_content = current_table.get("content", "")

                        # Для таблиц с разделителями (|)
                        if "|" in prev_content and "|" in curr_content:
                            # Посчитаем среднее количество столбцов
                            prev_lines = [line.count("|") for line in prev_content.split("\n") if "|" in line][:5]
                            curr_lines = [line.count("|") for line in curr_content.split("\n") if "|" in line][:5]

                            if prev_lines and curr_lines:
                                prev_avg = sum(prev_lines) / len(prev_lines)
                                curr_avg = sum(curr_lines) / len(curr_lines)

                                # Если структура таблиц схожа
                                if abs(prev_avg - curr_avg) <= 2:  # Допустимо небольшое различие
                                    is_continuation = True

                # Если это продолжение предыдущей таблицы, объединяем
                if is_continuation:
                    current_table["content"] += "\n\n" + chunk["content"]

                    # Собираем все страницы
                    if "all_pages" not in current_table:
                        current_table["all_pages"] = []

                    # Добавляем страницы из текущего чанка
                    if "pages" in chunk and chunk["pages"]:
                        if isinstance(chunk["pages"], list):
                            current_table["all_pages"].extend(chunk["pages"])
                        else:
                            current_table["all_pages"].append(chunk["pages"])
                    elif "page" in chunk and chunk["page"]:
                        current_table["all_pages"].append(chunk["page"])

                    # Убираем дубликаты и сортируем
                    current_table["all_pages"] = sorted(list(set(current_table["all_pages"])))

                    # Обновляем поле pages для совместимости
                    current_table["pages"] = current_table["all_pages"]
                else:
                    # Добавляем предыдущую таблицу и начинаем новую
                    if current_table:
                        processed_chunks.append(current_table)

                    current_table = chunk.copy()

                    # Инициализируем all_pages
                    if "pages" in chunk and chunk["pages"]:
                        current_table["all_pages"] = chunk["pages"] if isinstance(chunk["pages"], list) else [chunk["pages"]]
                    elif "page" in chunk and chunk["page"]:
                        current_table["all_pages"] = [chunk["page"]]
                    else:
                        current_table["all_pages"] = []

            else:
                # Обработка нетабличных элементов, которые могут быть продолжением таблицы
                if current_table:
                    curr_page = chunk.get("page")
                    prev_page = current_table.get("page")
                    content = chunk.get("content", "")

                    # Проверка на потенциальное продолжение таблицы
                    table_continuation = False

                    # Проверка 1: Элемент находится на следующей странице после таблицы
                    if curr_page and prev_page and curr_page - prev_page <= 2:  # Допускаем разрыв в 1-2 страницы
                        # Проверка на нумерацию, характерную для таблиц
                        import re

                        # Находим любые числовые пункты (например, "29.")
                        number_points = re.findall(r'^\s*(\d+)\.\s', content, re.MULTILINE)

                        if number_points:
                            # Извлекаем номера пунктов из текущей таблицы
                            table_numbers = []
                            table_content = current_table.get("content", "")
                            table_number_points = re.findall(r'^\s*(\d+)\.\s', table_content, re.MULTILINE)

                            if table_number_points:
                                table_numbers = [int(n) for n in table_number_points]
                                current_numbers = [int(n) for n in number_points]

                                # Проверяем, продолжается ли нумерация
                                if table_numbers and current_numbers:
                                    max_table_num = max(table_numbers)
                                    min_current_num = min(current_numbers)

                                    # Если нумерация последовательна или близка к последовательной
                                    if min_current_num > max_table_num and min_current_num - max_table_num <= 5:
                                        table_continuation = True

                        # Проверка 2: Анализ первой строки элемента
                        if not table_continuation:
                            first_line = content.strip().split('\n')[0] if '\n' in content else content.strip()

                            # Проверка, является ли текст продолжением предложения
                            # 1. Нет заглавной буквы в начале (продолжение предложения)
                            # 2. Начинается с предлога или союза (во многих языках)
                            # 3. Нет знаков препинания в начале

                            # Простая эвристика: если первая буква строчная и нет знаков препинания в начале
                            if first_line and not first_line[0].isupper() and not first_line[0] in ',.;:!?':
                                table_continuation = True

                    # Если это продолжение таблицы
                    if table_continuation:
                        # Объединяем с текущей таблицей
                        current_table["content"] += "\n\n" + content

                        # Обновляем all_pages
                        if "all_pages" not in current_table:
                            current_table["all_pages"] = []

                        if curr_page and curr_page not in current_table["all_pages"]:
                            current_table["all_pages"].append(curr_page)
                            current_table["all_pages"] = sorted(current_table["all_pages"])

                        # Пропускаем этот чанк в обработке
                        continue

                    # Если это не продолжение, завершаем таблицу
                    processed_chunks.append(current_table)
                    current_table = None

                # Добавляем all_pages для обычных чанков
                if "all_pages" not in chunk:
                    if "page" in chunk and chunk["page"]:
                        chunk["all_pages"] = [chunk["page"]]
                    else:
                        chunk["all_pages"] = []
                # Добавляем обычный чанк
                processed_chunks.append(chunk)

        # Добавляем последнюю таблицу, если она осталась
        if current_table:
            processed_chunks.append(current_table)

        return processed_chunks

    def group_semantic_chunks(self, chunks: List[Dict], min_length: int = 200) -> List[Dict]:
        """
        Объединяет все чанки с одной страницы с учетом продолжения таблиц
        """
        grouped_chunks = []
        current_page_chunks = []
        current_page = None
        last_table_caption = None

        # ОТЛАДКА: Логируем входные чанки
        logger.info(f"group_semantic_chunks: получено {len(chunks)} чанков")
        for i, chunk in enumerate(chunks):
            if chunk.get("type") == "section":
                logger.info(f"Входной чанк {i}: секция '{chunk.get('heading', 'Без заголовка')[:30]}...' "
                            f"имеет all_pages: {chunk.get('all_pages', [])}, page: {chunk.get('page')}")

        for i, chunk in enumerate(chunks):
            chunk_page = chunk.get("page")

            # Проверяем, является ли текущий чанк заголовком таблицы
            if chunk["type"] == "text" or chunk["type"] == "paragraph":
                if re.match(r'^Таблица\s*\d+[:.]\s*', chunk["content"], re.IGNORECASE):
                    last_table_caption = chunk["content"]
                    continue  # Пропускаем этот чанк, сохраняя заголовок для следующей таблицы

            # Если это таблица
            if chunk["type"] == "table":
                # Добавляем заголовок к таблице, если он есть
                if last_table_caption and not chunk.get("heading"):
                    chunk["heading"] = last_table_caption
                last_table_caption = None

            # Проверяем, не является ли текущий блок продолжением таблицы
            if grouped_chunks and chunk["type"] in ["text", "paragraph", "merged_page"]:
                prev_chunk = grouped_chunks[-1]

                # Если предыдущий чанк - таблица, и текущий находится на следующей странице
                if (prev_chunk["type"] == "table" and
                        chunk_page == prev_chunk.get("page", 0) + 1 and
                        is_likely_table_continuation(chunk["content"])):

                    # Объединяем с предыдущей таблицей
                    prev_chunk["content"] += "\n\n" + chunk["content"]

                    # Обновляем all_pages
                    if "all_pages" not in prev_chunk:
                        prev_chunk["all_pages"] = []
                    if chunk_page not in prev_chunk["all_pages"]:
                        prev_chunk["all_pages"].append(chunk_page)
                        prev_chunk["all_pages"] = sorted(prev_chunk["all_pages"])

                    if "pages" not in prev_chunk:
                        prev_chunk["pages"] = [prev_chunk.get("page")]
                    if chunk_page not in prev_chunk["pages"]:
                        prev_chunk["pages"].append(chunk_page)
                    prev_chunk["page"] = min(prev_chunk["pages"])  # Обновляем page до минимальной страницы
                    continue

            # ОТЛАДКА для секций
            if chunk.get("type") == "section":
                logger.info(f"Обрабатываем секцию '{chunk.get('heading', 'Без заголовка')[:30]}...' "
                            f"с all_pages: {chunk.get('all_pages', [])}, "
                            f"количество страниц: {len(chunk.get('all_pages', []))}")

            # ИСПРАВЛЕНИЕ: Проверяем, является ли чанк секцией с несколькими страницами
            if chunk.get("type") == "section" and chunk.get("all_pages") and len(chunk.get("all_pages", [])) > 1:
                logger.info(f"Секция с несколькими страницами обнаружена, добавляем без группировки")
                # Если есть накопленные чанки текущей страницы, сначала их обработаем
                if current_page_chunks:
                    grouped_chunks.append(self._merge_page_chunks(current_page_chunks))
                    current_page_chunks = []

                # Добавляем секцию как отдельный чанк, не группируя
                grouped_chunks.append(chunk)
                current_page = None  # Сбрасываем текущую страницу

            # Если страница изменилась и есть накопленные чанки
            elif chunk_page != current_page and current_page_chunks:
                logger.debug(f"Страница изменилась с {current_page} на {chunk_page}, группируем накопленные чанки")
                grouped_chunks.append(self._merge_page_chunks(current_page_chunks))
                current_page_chunks = [chunk]
                current_page = chunk_page

            # Добавляем чанк к текущей странице
            else:
                logger.debug(f"Добавляем чанк типа {chunk.get('type')} к странице {current_page}")
                current_page_chunks.append(chunk)
                current_page = chunk_page

        # Объединяем последнюю страницу
        if current_page_chunks:
            grouped_chunks.append(self._merge_page_chunks(current_page_chunks))

        # ОТЛАДКА: Логируем результат
        logger.info(f"group_semantic_chunks: результат {len(grouped_chunks)} чанков")
        for i, chunk in enumerate(grouped_chunks):
            if chunk.get("type") == "section":
                logger.info(f"Результат {i}: секция '{chunk.get('heading', 'Без заголовка')[:30]}...' "
                            f"имеет all_pages: {chunk.get('all_pages', [])}")

        return grouped_chunks

    def _merge_page_chunks(self, chunks: List[Dict]) -> Dict:
        """
        Объединяет чанки с одной страницы и собирает все номера страниц
        """
        if not chunks:
            return {}

        # ОТЛАДКА
        logger.debug(f"_merge_page_chunks: получено {len(chunks)} чанков для объединения")
        for i, chunk in enumerate(chunks):
            if chunk.get("type") == "section":
                logger.info(f"_merge_page_chunks: чанк {i} - секция '{chunk.get('heading', 'Без заголовка')[:30]}...' "
                            f"с all_pages: {chunk.get('all_pages', [])}")

        if len(chunks) == 1:
            chunk = chunks[0]
            # Если у чанка есть поле all_pages, используем его
            if "all_pages" in chunk and chunk["all_pages"]:
                pass  # Уже есть all_pages
            # Если у чанка есть поле pages (для таблиц), используем его
            elif "pages" in chunk and chunk["pages"]:
                chunk["all_pages"] = chunk["pages"]
            # Иначе используем поле page
            elif "page" in chunk and chunk["page"]:
                chunk["all_pages"] = [chunk["page"]]
            else:
                chunk["all_pages"] = []

            # ОТЛАДКА
            if chunk.get("type") == "section":
                logger.info(f"_merge_page_chunks: возвращаем единственный чанк - секцию "
                            f"'{chunk.get('heading', 'Без заголовка')[:30]}...' с all_pages: {chunk.get('all_pages', [])}")

            return chunk

        # Собираем все уникальные страницы из всех чанков
        all_pages = set()

        for chunk in chunks:
            # Проверяем поле all_pages
            if "all_pages" in chunk and chunk["all_pages"]:
                if isinstance(chunk["all_pages"], list):
                    all_pages.update(chunk["all_pages"])
                else:
                    all_pages.add(chunk["all_pages"])
            # Проверяем поле pages (для таблиц на нескольких страницах)
            elif "pages" in chunk and chunk["pages"]:
                if isinstance(chunk["pages"], list):
                    all_pages.update(chunk["pages"])
                else:
                    all_pages.add(chunk["pages"])
            # Проверяем поле page
            elif "page" in chunk and chunk["page"]:
                all_pages.add(chunk["page"])

        # Берем базовую информацию из первого чанка
        merged_chunk = {
            "content": "",
            "type": "merged_page",
            "page": chunks[0].get("page"),  # Оставляем для совместимости
            "all_pages": sorted(list(all_pages)),  # Список всех страниц
            "heading": None,
            "table_id": None
        }

        sections = []
        current_section = None

        for chunk in chunks:
            # Если это заголовок, начинаем новую секцию
            if chunk["type"] == "heading":
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "heading": chunk["content"],
                    "content": []
                }

            # Если уже есть секция, добавляем контент
            elif current_section:
                current_section["content"].append(chunk["content"])

            # Иначе добавляем как отдельный контент
            else:
                if chunk["content"].strip():
                    sections.append({
                        "heading": None,
                        "content": [chunk["content"]]
                    })

        # Добавляем последнюю секцию
        if current_section:
            sections.append(current_section)

        # Объединяем все секции
        content_parts = []
        for section in sections:
            if section["heading"]:
                content_parts.append(f"## {section['heading']}")
            content_parts.extend(section["content"])

        merged_chunk["content"] = "\n\n".join(content_parts)

        # ОТЛАДКА
        logger.info(f"_merge_page_chunks: результат объединения - тип: {merged_chunk['type']}, "
                    f"all_pages: {merged_chunk['all_pages']}")

        return merged_chunk

    def analyze_document(self, pdf_path: str) -> DocumentAnalysisResult:
        """
        Анализирует PDF документ и возвращает результаты в структурированном виде.

        Args:
            pdf_path: Путь к PDF файлу

        Returns:
            DocumentAnalysisResult: Результаты анализа документа
        """
        # Шаг 1: Извлекаем чанки
        chunks = self.extract_chunks(pdf_path)
        logger.info(f"Найдено {len(chunks)} начальных блоков")

        # Шаг 2: Обрабатываем таблицы
        processed_chunks = self.post_process_tables(chunks)
        logger.info(f"После обработки таблиц: {len(processed_chunks)} блоков")

        # Шаг 3: Группируем короткие блоки
        grouped_chunks = self.group_semantic_chunks(processed_chunks)
        logger.info(f"После группировки: {len(grouped_chunks)} финальных блоков")

        # Собираем статистику
        pages = set()
        content_types = {}

        for chunk in grouped_chunks:
            # Добавляем страницы
            if "all_pages" in chunk and chunk["all_pages"]:
                pages.update(chunk["all_pages"])
            elif chunk.get("pages"):
                pages.update(chunk["pages"])
            elif chunk.get("page"):
                pages.add(chunk["page"])

            # Подсчитываем типы контента
            chunk_type = chunk.get("type", "unknown")
            content_types[chunk_type] = content_types.get(chunk_type, 0) + 1

        # Создаем объекты SemanticChunk
        semantic_chunks = []
        for chunk in grouped_chunks:
            semantic_chunks.append(SemanticChunk(
                content=chunk["content"],
                type=chunk["type"],
                page=chunk.get("page"),
                heading=chunk.get("heading"),
                table_id=chunk.get("table_id"),
                pages=chunk.get("all_pages", chunk.get("pages")),
                section_path=None  # Можно добавить логику определения section_path
            ))

        # Формируем результат
        statistics = {
            "total_chunks": len(semantic_chunks),
            "pages": sorted(list(pages)),
            "total_pages": len(pages),
            "content_types": content_types
        }

        return DocumentAnalysisResult(
            chunks=semantic_chunks,
            document_path=pdf_path,
            statistics=statistics
        )

    def cleanup(self):
        """
        Освобождает ресурсы, выгружая Docling конвертер из памяти.
        """
        try:
            logger.info("Очистка ресурсов SemanticChunker...")

            # Освобождаем конвертер
            if self._converter_initialized:
                self._converter = None
                self._converter_initialized = False
                logger.info("Docling конвертер выгружен")

            # Принудительная сборка мусора
            import gc
            gc.collect()

            # Очистка CUDA кэша если используется GPU
            if self.use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        logger.info("CUDA кэш очищен после работы Docling")
                except Exception as e:
                    logger.warning(f"Не удалось очистить CUDA кэш: {e}")

        except Exception as e:
            logger.error(f"Ошибка при очистке ресурсов SemanticChunker: {e}")