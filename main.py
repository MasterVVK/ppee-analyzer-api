from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import redis.asyncio as redis
import json
import logging
import os
from contextlib import asynccontextmanager
import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import torch
import time
import psutil
import GPUtil

# Импорты из ppee_analyzer
from ppee_analyzer.vector_store import QdrantManager, BGEReranker, OllamaEmbeddings
from ppee_analyzer.semantic_chunker import SemanticChunker
from langchain_core.documents import Document

# Импорт локальных адаптеров
from app.adapters.llm_adapter import OllamaLLMProvider

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
redis_client = None
qdrant_manager = None
reranker = None
executor = ThreadPoolExecutor(max_workers=30)
reranker_semaphore = None
reranker_initialized = False  # Флаг инициализации ререйтера
active_indexing_tasks = 0  # Счетчик активных задач индексации
indexing_lock = None  # Локировка для счетчика
indexing_queue = None  # Очередь для задач индексации
indexing_semaphore = None  # Семафор для ограничения одной индексации

# Переменные для контроля нагрузки на эндпоинты
search_semaphore = asyncio.Semaphore(1)  # До 3 параллельных поисков
llm_semaphore = asyncio.Semaphore(1)  # Только 1 запрос к LLM одновременно
search_queue = asyncio.Queue()  # Очередь для поисковых запросов
llm_queue = asyncio.Queue()  # Очередь для LLM запросов

# Переменные для анализа
analysis_queue = None
analysis_semaphore = None
active_analysis_tasks = 0
analysis_lock = None
analysis_search_semaphore = None
analysis_llm_semaphore = None

# Настройки очистки памяти
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10s")  # Время хранения модели Ollama в памяти


# Pydantic модели
class IndexDocumentRequest(BaseModel):
    task_id: str
    application_id: str
    document_path: str
    document_id: Optional[str] = None  # Добавляем для совместимости
    delete_existing: bool = False
    metadata: Optional[Dict[str, Any]] = None  # Добавляем для передачи file_id


class SearchRequest(BaseModel):
    application_id: str
    query: str
    limit: int = 5
    use_reranker: bool = False
    rerank_limit: Optional[int] = None
    use_smart_search: bool = False
    vector_weight: float = 0.5
    text_weight: float = 0.5
    hybrid_threshold: int = 10


class ProcessQueryRequest(BaseModel):
    model_name: str
    prompt: str
    context: str
    parameters: Dict[str, Any]
    query: Optional[str] = None


class ChecklistItem(BaseModel):
    id: str
    name: str
    search_query: str
    search_limit: int = 3
    use_reranker: bool = False
    rerank_limit: Optional[int] = None
    llm_prompt_template: str
    llm_model: str
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1000
    context_length: int = 8192


class AnalyzeApplicationRequest(BaseModel):
    task_id: str
    application_id: str
    checklist_items: List[ChecklistItem]
    llm_params: Dict[str, Any] = {}


# Функции для управления памятью
def cleanup_gpu_memory():
    """Освобождает память GPU - только кэши, не выгружает модели"""
    try:
        # Явный вызов сборщика мусора Python
        gc.collect()

        # Если доступна CUDA, очищаем её кэш
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU кэши очищены")
    except Exception as e:
        logger.warning(f"Ошибка при очистке GPU памяти: {e}")


def calculate_content_weight(content_length: int) -> float:
    """
    Рассчитывает вес контента на основе его длины.

    Args:
        content_length: Длина контента в символах

    Returns:
        float: Вес от 0.1 до 1.0
    """
    if content_length < 10:
        return 0.1  # Почти пустые
    elif content_length < 50:
        return 0.5  # Короткие
    elif content_length < 200:
        return 0.8  # Средние
    else:
        return 1.0  # Полноценные


# Вспомогательные функции для поиска
def vector_search(application_id: str, query: str, limit: int,
                  use_reranker: bool, rerank_limit: Optional[int]) -> List[Dict]:
    """Векторный поиск с освобождением ресурсов"""
    try:
        # Получаем больше результатов для ререйтинга
        search_limit = rerank_limit if rerank_limit else (limit * 3 if use_reranker else limit)

        # Поиск
        docs = qdrant_manager.search(
            query=query,
            filter_dict={"application_id": application_id},
            k=search_limit
        )

        # Преобразуем результаты
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get('score', 0.0),
                "search_type": "vector"
            })

        # Ререйтинг (не загружаем модель, если не нужен)
        if use_reranker and results:
            # Получаем ререйтер через asyncio
            loop = asyncio.new_event_loop()
            current_reranker = loop.run_until_complete(get_reranker())
            loop.close()

            if current_reranker:
                try:
                    results = current_reranker.rerank(query, results, top_k=limit, text_key="text")
                finally:
                    # Ререйтер сам освобождает память в своем finally блоке
                    pass

        return results[:limit]
    except Exception as e:
        raise


def hybrid_search(application_id: str, query: str, limit: int,
                  vector_weight: float, text_weight: float, use_reranker: bool) -> List[Dict]:
    """Гибридный поиск с освобождением ресурсов"""
    try:
        # Векторный поиск (без ререйтинга на этом этапе)
        vector_results = vector_search(application_id, query, limit * 2, False, None)

        # Текстовый поиск через Qdrant
        from qdrant_client.http import models

        text_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.application_id",
                    match=models.MatchValue(value=application_id)
                ),
                models.FieldCondition(
                    key="page_content",
                    match=models.MatchText(text=query)
                )
            ]
        )

        text_results_raw = qdrant_manager.client.scroll(
            collection_name=qdrant_manager.collection_name,
            scroll_filter=text_filter,
            limit=limit * 2,
            with_payload=True,
            with_vectors=False
        )[0]

        # Преобразуем текстовые результаты
        text_results = []
        for i, point in enumerate(text_results_raw):
            if hasattr(point, 'payload'):
                text_results.append({
                    "text": point.payload.get("page_content", ""),
                    "metadata": point.payload.get("metadata", {}),
                    "score": 1.0 - (i * 0.05),
                    "search_type": "text"
                })

        # Объединяем результаты
        combined = combine_results(vector_results, text_results, vector_weight, text_weight, limit)

        # Ререйтинг
        if use_reranker and combined:
            # Получаем ререйтер через asyncio
            loop = asyncio.new_event_loop()
            current_reranker = loop.run_until_complete(get_reranker())
            loop.close()

            if current_reranker:
                try:
                    combined = current_reranker.rerank(query, combined, top_k=limit, text_key="text")
                finally:
                    # Ререйтер сам освобождает память в своем finally блоке
                    pass

        return combined[:limit]
    except Exception as e:
        raise


def combine_results(vector_results: List[Dict], text_results: List[Dict],
                    vector_weight: float, text_weight: float, limit: int) -> List[Dict]:
    """Объединяет результаты векторного и текстового поиска"""
    # Нормализуем веса
    total_weight = vector_weight + text_weight
    vector_weight = vector_weight / total_weight
    text_weight = text_weight / total_weight

    # Объединяем
    results_dict = {}

    for doc in vector_results:
        key = get_document_key(doc)
        results_dict[key] = {
            "doc": doc,
            "score": doc.get("score", 0.0) * vector_weight,
            "search_type": "hybrid"
        }

    for doc in text_results:
        key = get_document_key(doc)
        if key in results_dict:
            results_dict[key]["score"] += doc.get("score", 0.0) * text_weight
        else:
            results_dict[key] = {
                "doc": doc,
                "score": doc.get("score", 0.0) * text_weight,
                "search_type": "hybrid"
            }

    # Сортируем
    sorted_results = sorted(results_dict.values(), key=lambda x: x["score"], reverse=True)

    # Преобразуем обратно
    combined = []
    for item in sorted_results[:limit]:
        doc = item["doc"].copy()
        doc["score"] = item["score"]
        doc["search_type"] = "hybrid"
        combined.append(doc)

    return combined


def get_document_key(doc: Dict) -> str:
    """Создает уникальный ключ для документа"""
    metadata = doc.get("metadata", {})
    key_parts = []

    if "document_id" in metadata:
        key_parts.append(f"doc:{metadata['document_id']}")
    if "chunk_index" in metadata:
        key_parts.append(f"chunk:{metadata['chunk_index']}")

    return "|".join(key_parts) if key_parts else str(hash(doc.get("text", "")))


# Вспомогательные функции для анализа
def extract_value_from_response(response_data: Any, query: str) -> str:
    """Извлекает значение из ответа LLM"""
    # Если response_data - это словарь с полной информацией
    if isinstance(response_data, dict) and "response" in response_data:
        response = response_data["response"]
    else:
        # Обратная совместимость - если это просто строка
        response = str(response_data)

    lines = [line.strip() for line in response.split('\n') if line.strip()]

    # Ищем строку с результатом
    for line in lines:
        if line.startswith("РЕЗУЛЬТАТ:"):
            return line.replace("РЕЗУЛЬТАТ:", "").strip()

    # Ищем строку с двоеточием
    for line in lines:
        if ":" in line:
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()

    # Возвращаем последнюю строку
    return lines[-1] if lines else "Информация не найдена"


def calculate_confidence(response_data: Any) -> float:
    """Рассчитывает уверенность в ответе"""
    # Извлекаем текст ответа
    if isinstance(response_data, dict) and "response" in response_data:
        response = response_data["response"]
    else:
        response = str(response_data)

    uncertainty_phrases = [
        "возможно", "вероятно", "может быть", "предположительно",
        "не ясно", "не уверен", "не определено", "информация не найдена"
    ]

    confidence = 0.8
    response_lower = response.lower()

    for phrase in uncertainty_phrases:
        if phrase in response_lower:
            confidence -= 0.1

    return max(0.1, min(confidence, 1.0))


# Асинхронные функции
async def periodic_cleanup():
    """Периодически очищает неиспользуемые кэши GPU"""
    while True:
        await asyncio.sleep(300)  # Каждые 5 минут

        # Очищаем только если нет активных задач
        if active_indexing_tasks == 0:
            cleanup_gpu_memory()
            logger.info("Выполнена периодическая очистка GPU кэшей")


async def indexing_queue_worker():
    """Воркер для обработки очереди индексации - обрабатывает только одну задачу за раз"""
    logger.info("Запущен обработчик очереди индексации")

    while True:
        try:
            # Ждем задачу из очереди
            request = await indexing_queue.get()

            # Обрабатываем задачу с семафором (только одна за раз)
            async with indexing_semaphore:
                logger.info(f"Начало обработки задачи индексации: {request.task_id}")
                await index_document_task_worker(request)
                logger.info(f"Завершена обработка задачи индексации: {request.task_id}")

            # Помечаем задачу как выполненную
            indexing_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Остановка обработчика очереди индексации")
            break
        except Exception as e:
            logger.error(f"Ошибка в обработчике очереди индексации: {e}")
            await asyncio.sleep(1)  # Небольшая пауза при ошибке


async def analysis_queue_worker():
    """Воркер для обработки очереди анализа - обрабатывает только один анализ за раз"""
    logger.info("Запущен обработчик очереди анализа")

    while True:
        try:
            # Ждем задачу из очереди
            request = await analysis_queue.get()

            # Обрабатываем задачу с семафором (только одна за раз)
            async with analysis_semaphore:
                logger.info(f"Начало анализа: {request.task_id}")
                await analyze_application_task_worker(request)
                logger.info(f"Завершен анализ: {request.task_id}")

            # Помечаем задачу как выполненную
            analysis_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Остановка обработчика очереди анализа")
            break
        except Exception as e:
            logger.error(f"Ошибка в обработчике очереди анализа: {e}")
            await asyncio.sleep(1)  # Небольшая пауза при ошибке


async def search_queue_worker():
    """Воркер для обработки очереди поисковых запросов"""
    logger.info("Запущен обработчик очереди поиска")

    while True:
        try:
            # Ждем задачу из очереди
            task_data = await search_queue.get()

            # Обрабатываем задачу с семафором
            async with search_semaphore:
                await _process_search_request(task_data)

            # Помечаем задачу как выполненную
            search_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Остановка обработчика очереди поиска")
            break
        except Exception as e:
            logger.error(f"Ошибка в обработчике очереди поиска: {e}")
            # Устанавливаем ошибку в future если она есть
            if 'future' in task_data and not task_data['future'].done():
                task_data['future'].set_exception(e)


async def llm_queue_worker():
    """Воркер для обработки очереди LLM запросов"""
    logger.info("Запущен обработчик очереди LLM")

    while True:
        try:
            # Ждем задачу из очереди
            task_data = await llm_queue.get()

            # Обрабатываем задачу с семафором
            async with llm_semaphore:
                await _process_llm_request(task_data)

            # Помечаем задачу как выполненную
            llm_queue.task_done()

        except asyncio.CancelledError:
            logger.info("Остановка обработчика очереди LLM")
            break
        except Exception as e:
            logger.error(f"Ошибка в обработчике очереди LLM: {e}")
            # Устанавливаем ошибку в future если она есть
            if 'future' in task_data and not task_data['future'].done():
                task_data['future'].set_exception(e)


async def get_reranker():
    """Ленивая инициализация ререйтера при первом использовании"""
    global reranker, reranker_initialized

    if not reranker_initialized:
        async with reranker_semaphore:  # Блокируем для потокобезопасности
            if not reranker_initialized:  # Двойная проверка
                try:
                    logger.info("Инициализация ререйтера при первом использовании...")
                    loop = asyncio.get_event_loop()
                    reranker = await loop.run_in_executor(
                        executor,
                        lambda: BGEReranker(
                            model_name="BAAI/bge-reranker-v2-m3",
                            device="cuda",
                            batch_size=1,  # Используем batch_size=1 для экономии памяти
                            max_length=8192,
                            min_vram_mb=500
                        )
                    )
                    reranker_initialized = True
                    logger.info("Ререйтер успешно инициализирован")
                except Exception as e:
                    logger.warning(f"Не удалось инициализировать ререйтер: {e}")
                    reranker = None
                    reranker_initialized = True  # Помечаем как инициализированный, чтобы не пытаться снова

    return reranker


async def update_task_status(task_id: str, status: str, progress: int = 0,
                             stage: str = "", message: str = "", result: Any = None):
    """Асинхронно обновляет статус задачи в Redis"""
    task_data = {
        "status": status,
        "progress": progress,
        "stage": stage,
        "message": message
    }
    if result is not None:
        task_data["result"] = result

    await redis_client.setex(f"task:{task_id}", 7200, json.dumps(task_data))
    logger.info(f"Task {task_id}: {status} - {progress}% [{stage}] {message}")


async def process_document_with_semantic_chunker(document_path: str, application_id: str,
                                                 additional_metadata: Optional[Dict[str, Any]] = None) -> List[
    Document]:
    """Асинхронно обрабатывает документ с использованием семантического чанкера"""
    loop = asyncio.get_event_loop()

    def _process():
        chunker = None
        try:
            # Перед созданием chunker'а очищаем память и устанавливаем устройство
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_device(0)  # Явно устанавливаем устройство

            chunker = SemanticChunker(use_gpu=True)

            # Извлекаем и обрабатываем чанки
            chunks = chunker.extract_chunks(document_path)
            processed_chunks = chunker.post_process_tables(chunks)
            grouped_chunks = chunker.group_semantic_chunks(processed_chunks)

            # Добавляем отладочную информацию
            logger.info(f"Обработка {len(grouped_chunks)} чанков для документа {document_path}")

            # Преобразуем в Document объекты
            documents = []

            # Используем document_id из additional_metadata если передан
            document_id = None
            if additional_metadata and 'document_id' in additional_metadata:
                document_id = additional_metadata['document_id']
            else:
                document_id = f"doc_{os.path.basename(document_path).replace(' ', '_').replace('.', '_')}"

            document_name = os.path.basename(document_path)

            for i, chunk in enumerate(grouped_chunks):
                # Отладка для секций
                if chunk.get("type") == "section":
                    logger.debug(f"Секция '{chunk.get('heading', 'Без заголовка')[:30]}...' "
                                 f"имеет all_pages: {chunk.get('all_pages', [])}")

                # Получаем контент
                content = chunk.get("content", "")
                content_length = len(content.strip())

                metadata = {
                    "application_id": application_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "content_type": chunk.get("type", "unknown"),
                    "chunk_index": i,
                    "section": chunk.get("heading", "Не определено"),
                    # Добавляем метки и вес контента
                    "is_empty": content_length < 10,
                    "content_length": content_length,
                    "content_weight": calculate_content_weight(content_length)  # Новое поле
                }

                # Добавляем дополнительные метаданные (file_id, index_session_id и т.д.)
                if additional_metadata:
                    metadata.update(additional_metadata)

                # Обработка страниц
                pages_list = []

                # Сначала проверяем all_pages - это основное поле для ВСЕХ страниц
                if "all_pages" in chunk and chunk["all_pages"]:
                    pages_list = chunk["all_pages"]
                    logger.debug(f"Чанк {i} ({chunk.get('type')}): используем all_pages = {pages_list}")
                # Затем проверяем pages (для совместимости)
                elif "pages" in chunk and chunk["pages"]:
                    pages_list = chunk["pages"] if isinstance(chunk["pages"], list) else [chunk["pages"]]
                    logger.debug(f"Чанк {i} ({chunk.get('type')}): используем pages = {pages_list}")
                # В крайнем случае используем page
                elif "page" in chunk and chunk["page"]:
                    pages_list = [chunk["page"]]
                    logger.debug(f"Чанк {i} ({chunk.get('type')}): используем page = {pages_list}")

                # Убеждаемся, что pages_list - это список
                if not isinstance(pages_list, list):
                    pages_list = [pages_list] if pages_list else []

                # Сохраняем страницы в метаданные
                if pages_list:
                    # Сохраняем как строку через запятую для удобства поиска
                    metadata["page_number"] = ",".join(map(str, sorted(pages_list)))
                    # Также сохраняем как список для удобства обработки
                    metadata["page_numbers"] = sorted(pages_list)
                else:
                    metadata["page_number"] = ""
                    metadata["page_numbers"] = []

                documents.append(Document(
                    page_content=content,  # Используем оригинальный контент как есть
                    metadata=metadata
                ))

            return documents

        except RuntimeError as e:
            if "meta tensor" in str(e):
                logger.error(f"Ошибка meta tensor, попытка использовать CPU: {str(e)}")
                # Пересоздаем chunker с CPU
                if chunker:
                    if hasattr(chunker, '_converter'):
                        chunker._converter = None
                        chunker._converter_initialized = False
                    del chunker

                # Повторная попытка с CPU
                chunker = SemanticChunker(use_gpu=False)
                chunks = chunker.extract_chunks(document_path)
                processed_chunks = chunker.post_process_tables(chunks)
                grouped_chunks = chunker.group_semantic_chunks(processed_chunks)

                # Преобразуем в Document объекты (повторяем ту же логику)
                documents = []

                # Используем document_id из additional_metadata если передан
                document_id = None
                if additional_metadata and 'document_id' in additional_metadata:
                    document_id = additional_metadata['document_id']
                else:
                    document_id = f"doc_{os.path.basename(document_path).replace(' ', '_').replace('.', '_')}"

                document_name = os.path.basename(document_path)

                for i, chunk in enumerate(grouped_chunks):
                    content = chunk.get("content", "")
                    content_length = len(content.strip())

                    metadata = {
                        "application_id": application_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "content_type": chunk.get("type", "unknown"),
                        "chunk_index": i,
                        "section": chunk.get("heading", "Не определено"),
                        "is_empty": content_length < 10,
                        "content_length": content_length,
                        "content_weight": calculate_content_weight(content_length)
                    }

                    # Добавляем дополнительные метаданные
                    if additional_metadata:
                        metadata.update(additional_metadata)

                    # Используем ту же логику для страниц
                    pages_list = []

                    if "all_pages" in chunk and chunk["all_pages"]:
                        pages_list = chunk["all_pages"]
                    elif "pages" in chunk and chunk["pages"]:
                        pages_list = chunk["pages"] if isinstance(chunk["pages"], list) else [chunk["pages"]]
                    elif "page" in chunk and chunk["page"]:
                        pages_list = [chunk["page"]]

                    if not isinstance(pages_list, list):
                        pages_list = [pages_list] if pages_list else []

                    if pages_list:
                        metadata["page_number"] = ",".join(map(str, sorted(pages_list)))
                        metadata["page_numbers"] = sorted(pages_list)
                    else:
                        metadata["page_number"] = ""
                        metadata["page_numbers"] = []

                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))

                return documents
            else:
                raise
        finally:
            # Очищаем chunker после использования
            if chunker:
                # Освобождаем docling converter
                if hasattr(chunker, '_converter'):
                    chunker._converter = None
                    chunker._converter_initialized = False
                del chunker

                # Принудительная сборка мусора
                import gc
                gc.collect()

                # Очищаем только GPU кэши (не модели!)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info("SemanticChunker освобожден из памяти")

    return await loop.run_in_executor(executor, _process)


async def index_document_task_worker(request: IndexDocumentRequest):
    """Воркер для выполнения индексации - вызывается из очереди"""
    global active_indexing_tasks

    # Увеличиваем счетчик активных задач
    async with indexing_lock:
        active_indexing_tasks += 1
        logger.info(f"Начата индексация. Активных задач: {active_indexing_tasks}")

    try:
        await update_task_status(request.task_id, "PROGRESS", 5, "prepare", "Подготовка к индексации...")

        # Проверяем файл
        if not os.path.exists(request.document_path):
            raise FileNotFoundError(f"Файл не найден: {request.document_path}")

        await update_task_status(request.task_id, "PROGRESS", 10, "convert", "Конвертация документа...")

        # Обрабатываем документ с передачей метаданных
        chunks = await process_document_with_semantic_chunker(
            request.document_path,
            request.application_id,
            request.metadata  # Передаем дополнительные метаданные
        )

        # НОВОЕ: Добавляем этап "split" после обработки
        await update_task_status(request.task_id, "PROGRESS", 30, "split", f"Разделение на {len(chunks)} фрагментов...")

        # Удаляем старые данные если нужно
        if request.delete_existing:
            loop = asyncio.get_event_loop()

            # Если есть file_id в метаданных, удаляем по нему
            if request.metadata and 'file_id' in request.metadata:
                # Удаляем чанки конкретного файла
                from qdrant_client.http import models

                delete_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.application_id",
                            match=models.MatchValue(value=request.application_id)
                        ),
                        models.FieldCondition(
                            key="metadata.file_id",
                            match=models.MatchValue(value=request.metadata['file_id'])
                        )
                    ]
                )

                # Получаем точки для удаления
                scroll_result = await loop.run_in_executor(
                    executor,
                    lambda: qdrant_manager.client.scroll(
                        collection_name=qdrant_manager.collection_name,
                        scroll_filter=delete_filter,
                        limit=10000,
                        with_payload=False,
                        with_vectors=False
                    )
                )

                points_to_delete = [point.id for point in scroll_result[0]]

                if points_to_delete:
                    await loop.run_in_executor(
                        executor,
                        lambda: qdrant_manager.client.delete(
                            collection_name=qdrant_manager.collection_name,
                            points_selector=models.PointIdsList(points=points_to_delete)
                        )
                    )
                    logger.info(f"Удалено {len(points_to_delete)} старые чанков файла")
            else:
                # Старый способ - удаляем все данные заявки
                await loop.run_in_executor(executor, qdrant_manager.delete_application, request.application_id)

        # НОВОЕ: Обновляем прогресс перед индексацией
        await update_task_status(request.task_id, "PROGRESS", 50, "index", f"Индексация {len(chunks)} фрагментов...")

        # Индексируем пакетами асинхронно
        total_chunks = len(chunks)
        batch_size = 20

        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]

            # Добавляем документы в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, qdrant_manager.add_documents, batch)

            # НОВОЕ: Корректируем расчет прогресса и этапа
            progress = 50 + int(40 * (end_idx / total_chunks))  # От 50% до 90%

            # Определяем этап в зависимости от прогресса
            if progress < 90:
                stage = "index"
            else:
                stage = "complete"

            await update_task_status(request.task_id, "PROGRESS", progress, stage,
                                     f"Индексация: {end_idx}/{total_chunks}...")

        # НОВОЕ: Финальное обновление
        await update_task_status(request.task_id, "PROGRESS", 95, "complete", "Завершение индексации...")

        # Успешное завершение
        result = {
            "application_id": request.application_id,
            "document_path": request.document_path,
            "total_chunks": total_chunks,
            "status": "success"
        }

        await update_task_status(request.task_id, "SUCCESS", 100, "complete",
                                 "Индексация завершена", result)

    except Exception as e:
        logger.exception(f"Ошибка индексации: {e}")
        await update_task_status(request.task_id, "FAILURE", 0, "error", str(e))
    finally:
        # Уменьшаем счетчик активных задач
        async with indexing_lock:
            active_indexing_tasks -= 1
            logger.info(f"Индексация завершена. Активных задач: {active_indexing_tasks}")

        # Убираем агрессивную очистку, оставляем только легкую очистку GPU кэшей
        cleanup_gpu_memory()


async def _process_search_request(task_data: dict):
    """Внутренняя функция для обработки поискового запроса"""
    request = task_data['request']
    future = task_data['future']

    try:
        loop = asyncio.get_event_loop()

        # Выполняем поиск в отдельном потоке
        if request.use_smart_search:
            # Умный поиск
            if len(request.query) < request.hybrid_threshold:
                # Гибридный поиск
                results = await loop.run_in_executor(
                    executor,
                    hybrid_search,
                    request.application_id,
                    request.query,
                    request.limit,
                    request.vector_weight,
                    request.text_weight,
                    request.use_reranker
                )
            else:
                # Векторный поиск
                results = await loop.run_in_executor(
                    executor,
                    vector_search,
                    request.application_id,
                    request.query,
                    request.limit,
                    request.use_reranker,
                    request.rerank_limit
                )
        else:
            # Обычный векторный поиск
            results = await loop.run_in_executor(
                executor,
                vector_search,
                request.application_id,
                request.query,
                request.limit,
                request.use_reranker,
                request.rerank_limit
            )

        # Устанавливаем результат
        future.set_result({"status": "success", "results": results})

    except Exception as e:
        # Устанавливаем исключение
        future.set_exception(e)


async def _process_llm_request(task_data: dict):
    """Внутренняя функция для обработки LLM запроса"""
    request = task_data['request']
    future = task_data['future']

    try:
        loop = asyncio.get_event_loop()

        def _process():
            llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")
            return llm_provider.process_query(
                model_name=request.model_name,
                prompt=request.prompt,
                context=request.context,
                parameters=request.parameters,
                query=request.query
            )

        # Получаем полный ответ с информацией о токенах
        llm_result = await loop.run_in_executor(executor, _process)

        # Устанавливаем результат с полной информацией
        future.set_result({
            "status": "success",
            "response": llm_result.get("response", ""),
            "tokens": {
                "prompt_tokens": llm_result.get("prompt_tokens", 0),
                "completion_tokens": llm_result.get("completion_tokens", 0),
                "total_tokens": llm_result.get("total_tokens", 0)
            },
            "model": llm_result.get("model", request.model_name)
        })

    except Exception as e:
        # Устанавливаем исключение
        future.set_exception(e)


async def process_llm_query_async(model_name: str, prompt: str, context: str,
                                  parameters: Dict[str, Any], query: Optional[str] = None):
    """Асинхронная обработка запроса через LLM с возвратом информации о токенах"""
    loop = asyncio.get_event_loop()

    def _process():
        llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")
        # Теперь возвращаем полный ответ с информацией о токенах
        return llm_provider.process_query(
            model_name=model_name,
            prompt=prompt,
            context=context,
            parameters=parameters,
            query=query
        )

    return await loop.run_in_executor(executor, _process)


async def analyze_application_task_worker(request: AnalyzeApplicationRequest):
    """Воркер для выполнения анализа - вызывается из очереди"""
    global active_analysis_tasks

    # Увеличиваем счетчик активных задач
    async with analysis_lock:
        active_analysis_tasks += 1
        logger.info(f"Начат анализ. Активных анализов: {active_analysis_tasks}")
    try:
        await update_task_status(request.task_id, "PROGRESS", 10, "prepare", "Инициализация анализа...")

        loop = asyncio.get_event_loop()
        processed_count = 0
        error_count = 0
        total_items = len(request.checklist_items)

        results = []

        # Получаем настройки умного поиска из llm_params
        use_smart_search = request.llm_params.get('use_smart_search', True)
        hybrid_threshold = request.llm_params.get('hybrid_threshold', 10)

        for i, item in enumerate(request.checklist_items):
            try:
                progress = 15 + int(75 * (i / total_items))
                await update_task_status(request.task_id, "PROGRESS", progress, "analyze",
                                         f"Анализ параметра {i + 1}/{total_items}: {item['name']}")

                # Определяем метод поиска на основе длины запроса
                query = item['search_query']
                search_method = "vector"

                # Выполняем поиск С СЕМАФОРОМ
                async with analysis_search_semaphore:
                    if use_smart_search:
                        if len(query) < hybrid_threshold:
                            logger.info(f"Используем гибридный поиск для '{query}' (<{hybrid_threshold} символов)")
                            search_method = "hybrid"

                            # Выполняем гибридный поиск
                            search_results = await loop.run_in_executor(
                                executor,
                                hybrid_search,
                                request.application_id,
                                query,
                                item.get('search_limit', 3),
                                0.5,  # vector_weight
                                0.5,  # text_weight
                                item.get('use_reranker', False)
                            )
                        else:
                            logger.info(f"Используем векторный поиск для '{query}' (>={hybrid_threshold} символов)")

                            # Выполняем векторный поиск
                            search_results = await loop.run_in_executor(
                                executor,
                                vector_search,
                                request.application_id,
                                query,
                                item.get('search_limit', 3),
                                item.get('use_reranker', False),
                                item.get('rerank_limit', 10) if item.get('use_reranker', False) else None
                            )
                    else:
                        # Обычный векторный поиск
                        search_results = await loop.run_in_executor(
                            executor,
                            vector_search,
                            request.application_id,
                            query,
                            item.get('search_limit', 3),
                            item.get('use_reranker', False),
                            item.get('rerank_limit', 10) if item.get('use_reranker', False) else None
                        )

                # Преобразуем результаты в формат Document для LLM
                search_documents = []
                for res in search_results:
                    doc = Document(
                        page_content=res.get('text', ''),
                        metadata=res.get('metadata', {})
                    )
                    search_documents.append(doc)

                # Обработка через LLM С СЕМАФОРОМ
                if search_documents:
                    # Форматируем контекст
                    context = "\n\n".join([doc.page_content for doc in search_documents])

                    # Формируем полный промпт здесь
                    full_prompt = item['llm_prompt_template'].replace(
                        "{query}", query
                    ).replace(
                        "{context}", context
                    )

                    # Вызываем LLM асинхронно с полным промптом
                    async with analysis_llm_semaphore:
                        # Теперь получаем полный ответ с информацией о токенах
                        llm_result = await process_llm_query_async(
                            item['llm_model'],
                            full_prompt,  # Передаем полный промпт
                            "",  # Контекст уже включен в промпт
                            {
                                'temperature': item.get('llm_temperature', 0.1),
                                'max_tokens': item.get('llm_max_tokens', 1000),
                                'context_length': item.get('context_length', 8192)
                            },
                            query
                        )

                    # Теперь llm_result - это словарь с полной информацией
                    llm_response = llm_result.get("response", "")
                    tokens_info = {
                        "prompt_tokens": llm_result.get("prompt_tokens", 0),
                        "completion_tokens": llm_result.get("completion_tokens", 0),
                        "total_tokens": llm_result.get("total_tokens", 0)
                    }

                    # Извлекаем значение из текста ответа (передаем только текст)
                    value = extract_value_from_response(llm_response, query)
                    confidence = calculate_confidence(llm_response)

                    results.append({
                        "parameter_id": item['id'],
                        "value": value,
                        "confidence": confidence,
                        "search_results": [{"text": res.get('text', ''),
                                            "metadata": res.get('metadata', {}),
                                            "score": res.get('score', 0.0)}
                                           for res in search_results],
                        "search_method": search_method,
                        "llm_request": {
                            'prompt': full_prompt,  # Сохраняем полный промпт
                            'prompt_template': item['llm_prompt_template'],  # Шаблон для справки
                            'model': item['llm_model'],
                            'temperature': item.get('llm_temperature', 0.1),
                            'max_tokens': item.get('llm_max_tokens', 1000),
                            'context_length': item.get('context_length', 8192),
                            'response': llm_response,  # Полный ответ LLM
                            'search_query': query,
                            'search_method': search_method,
                            'context': context,  # Можно сохранить и контекст для отладки
                            # НОВОЕ: Добавляем информацию о токенах
                            'tokens': tokens_info,
                            'prompt_tokens': tokens_info['prompt_tokens'],
                            'completion_tokens': tokens_info['completion_tokens'],
                            'total_tokens': tokens_info['total_tokens']
                        }
                    })
                    processed_count += 1

                    # НОВОЕ: Сохраняем промежуточные результаты в Redis после каждого параметра
                    await redis_client.setex(
                        f"task_results:{request.task_id}",
                        3600,  # TTL 1 час
                        json.dumps({"results": results})
                    )
                    logger.info(f"Сохранены промежуточные результаты в Redis: {len(results)} параметров")

                else:
                    results.append({
                        "parameter_id": item['id'],
                        "value": "Информация не найдена",
                        "confidence": 0.0,
                        "search_results": [],
                        "search_method": search_method,
                        "llm_request": {
                            'error': 'Не найдено результатов поиска',
                            'search_method': search_method,
                            'model': item['llm_model'],
                            'search_query': query,
                            # Добавляем пустую информацию о токенах для консистентности
                            'tokens': {
                                'prompt_tokens': 0,
                                'completion_tokens': 0,
                                'total_tokens': 0
                            },
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0
                        }
                    })

                # После каждых 10 параметров освобождаем кэши
                if (i + 1) % 10 == 0:
                    cleanup_gpu_memory()

            except Exception as e:
                logger.error(f"Ошибка при обработке параметра {item['id']}: {e}")
                error_count += 1
                results.append({
                    "parameter_id": item['id'],
                    "value": "Ошибка обработки",
                    "confidence": 0.0,
                    "search_results": [],
                    "llm_request": {
                        'error': str(e),
                        'model': item.get('llm_model', 'unknown'),
                        'tokens': {
                            'prompt_tokens': 0,
                            'completion_tokens': 0,
                            'total_tokens': 0
                        }
                    }
                })

        # Освобождаем кэши после завершения анализа
        cleanup_gpu_memory()

        # Завершение - сохраняем финальные результаты
        result = {
            "status": "success",
            "processed": processed_count,
            "errors": error_count,
            "total": total_items,
            "results": results
        }

        # Сохраняем финальные результаты в Redis
        await redis_client.setex(
            f"task_results:{request.task_id}",
            3600,  # TTL 1 час
            json.dumps({"results": results})
        )

        await update_task_status(request.task_id, "SUCCESS", 100, "complete",
                                 "Анализ завершен", result)

    except Exception as e:
        logger.exception(f"Ошибка анализа: {e}")
        cleanup_gpu_memory()  # Освобождаем кэши при ошибке
        await update_task_status(request.task_id, "FAILURE", 0, "error", str(e))
    finally:
        # Уменьшаем счетчик активных задач
        async with analysis_lock:
            active_analysis_tasks -= 1
            logger.info(f"Анализ завершен. Активных анализов: {active_analysis_tasks}")

        # Чистка GPU
        cleanup_gpu_memory()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, qdrant_manager, reranker_semaphore, indexing_lock, indexing_queue, indexing_semaphore
    global analysis_queue, analysis_semaphore, analysis_lock, analysis_search_semaphore, analysis_llm_semaphore

    # Инициализация семафора для ререйтера
    reranker_semaphore = asyncio.Semaphore(1)

    # Инициализация блокировки для счетчика индексаций
    indexing_lock = asyncio.Lock()

    # Инициализация очереди для задач индексации
    indexing_queue = asyncio.Queue()

    # Семафор для ограничения одной индексации за раз
    indexing_semaphore = asyncio.Semaphore(1)

    # Инициализация для анализа
    analysis_queue = asyncio.Queue()
    analysis_semaphore = asyncio.Semaphore(1)  # Один анализ за раз
    analysis_lock = asyncio.Lock()
    analysis_search_semaphore = asyncio.Semaphore(3)  # До 3 параллельных поисков в анализе
    analysis_llm_semaphore = asyncio.Semaphore(1)  # Один LLM запрос за раз в анализе

    # Инициализация Redis с асинхронным клиентом
    redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)
    logger.info("Redis клиент инициализирован")

    # Инициализация QdrantManager БЕЗ загрузки моделей
    loop = asyncio.get_event_loop()
    qdrant_manager = await loop.run_in_executor(
        executor,
        lambda: QdrantManager(
            collection_name="ppee_applications",
            host="localhost",
            port=6333,
            embeddings_type="ollama",
            model_name="bge-m3",
            ollama_url="http://localhost:11434",
            check_availability=False,  # ВАЖНО: Отключаем проверку при старте
            ollama_keep_alive=OLLAMA_KEEP_ALIVE  # Время хранения модели в памяти из переменной окружения
        )
    )
    logger.info("QdrantManager инициализирован")

    # НОВОЕ: Проверка и создание коллекции Qdrant
    logger.info("Проверка и создание коллекции Qdrant...")
    try:
        # Получаем список существующих коллекций
        collections_response = await loop.run_in_executor(
            executor,
            lambda: qdrant_manager.client.get_collections()
        )

        existing_collections = [c.name for c in collections_response.collections]
        logger.info(f"Существующие коллекции: {existing_collections}")

        # Проверяем, существует ли наша коллекция
        if qdrant_manager.collection_name not in existing_collections:
            logger.info(f"Коллекция '{qdrant_manager.collection_name}' не найдена. Создаем...")

            # Создаем коллекцию
            from qdrant_client.http import models
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_collection(
                    collection_name=qdrant_manager.collection_name,
                    vectors_config=models.VectorParams(
                        size=qdrant_manager.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
            )
            logger.info(f"Коллекция '{qdrant_manager.collection_name}' создана")

            # Создаем индексы для оптимизации поиска
            logger.info("Создание индексов...")

            # Индекс для application_id
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_payload_index(
                    collection_name=qdrant_manager.collection_name,
                    field_name="metadata.application_id",
                    field_schema="keyword"
                )
            )

            # Индекс для content_type
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_payload_index(
                    collection_name=qdrant_manager.collection_name,
                    field_name="metadata.content_type",
                    field_schema="keyword"
                )
            )

            # Полнотекстовый индекс для page_content
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_payload_index(
                    collection_name=qdrant_manager.collection_name,
                    field_name="page_content",
                    field_schema="text"
                )
            )

            # НОВЫЕ ИНДЕКСЫ ДЛЯ ФИЛЬТРАЦИИ ПУСТЫХ ЧАНКОВ
            # Индекс для фильтрации пустых чанков
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_payload_index(
                    collection_name=qdrant_manager.collection_name,
                    field_name="metadata.is_empty",
                    field_schema="bool"
                )
            )
            logger.info("Индекс для metadata.is_empty создан")

            # Индекс для фильтрации по длине контента
            await loop.run_in_executor(
                executor,
                lambda: qdrant_manager.client.create_payload_index(
                    collection_name=qdrant_manager.collection_name,
                    field_name="metadata.content_length",
                    field_schema="integer"
                )
            )
            logger.info("Индекс для metadata.content_length создан")

            logger.info("Все индексы созданы успешно")

        else:
            logger.info(f"Коллекция '{qdrant_manager.collection_name}' уже существует")

            # Проверяем и создаем недостающие индексы
            try:
                collection_info = await loop.run_in_executor(
                    executor,
                    lambda: qdrant_manager.client.get_collection(qdrant_manager.collection_name)
                )

                # Список необходимых индексов
                required_indices = {
                    "metadata.application_id": "keyword",
                    "metadata.content_type": "keyword",
                    "page_content": "text",
                    "metadata.is_empty": "bool",
                    "metadata.content_length": "integer",
                    "metadata.content_weight": "float"
                }

                # Проверяем наличие полнотекстового индекса
                if hasattr(collection_info, 'payload_schema'):
                    existing_indices = set(
                        collection_info.payload_schema.keys()) if collection_info.payload_schema else set()

                    # Создаем недостающие индексы
                    for field_name, field_schema in required_indices.items():
                        if field_name not in existing_indices:
                            logger.info(f"Создание недостающего индекса для {field_name}...")
                            try:
                                await loop.run_in_executor(
                                    executor,
                                    lambda fn=field_name, fs=field_schema: qdrant_manager.client.create_payload_index(
                                        collection_name=qdrant_manager.collection_name,
                                        field_name=fn,
                                        field_schema=fs
                                    )
                                )
                                logger.info(f"Индекс для {field_name} создан")
                            except Exception as idx_error:
                                logger.warning(f"Не удалось создать индекс для {field_name}: {idx_error}")

            except Exception as e:
                logger.warning(f"Не удалось проверить индексы: {e}")

        # Получаем информацию о коллекции
        collection_info = await loop.run_in_executor(
            executor,
            lambda: qdrant_manager.client.get_collection(qdrant_manager.collection_name)
        )

        logger.info(f"Коллекция готова к работе. Статус: {collection_info.status}, "
                    f"Векторов: {collection_info.vectors_count}")

    except Exception as e:
        logger.error(f"Критическая ошибка при создании коллекции: {e}")

    # НЕ инициализируем ререйтер при старте
    logger.info("Сервис запущен. Модели будут загружены при первом использовании.")

    # Запускаем периодическую очистку памяти
    cleanup_task = asyncio.create_task(periodic_cleanup())

    # Запускаем обработчик очереди индексации
    indexing_worker_task = asyncio.create_task(indexing_queue_worker())

    # Запускаем обработчик очереди анализа
    analysis_worker_task = asyncio.create_task(analysis_queue_worker())

    # Запускаем обработчики очередей для эндпоинтов
    search_worker_task = asyncio.create_task(search_queue_worker())
    llm_worker_task = asyncio.create_task(llm_queue_worker())

    yield

    # Shutdown
    logger.info("Начало остановки сервиса...")

    # Отменяем фоновые задачи
    cleanup_task.cancel()
    indexing_worker_task.cancel()
    analysis_worker_task.cancel()
    search_worker_task.cancel()
    llm_worker_task.cancel()

    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    try:
        await indexing_worker_task
    except asyncio.CancelledError:
        pass

    try:
        await analysis_worker_task
    except asyncio.CancelledError:
        pass

    try:
        await search_worker_task
    except asyncio.CancelledError:
        pass

    try:
        await llm_worker_task
    except asyncio.CancelledError:
        pass

    # Ожидаем завершения всех задач в очереди
    if indexing_queue.qsize() > 0:
        logger.info(f"Ожидание завершения {indexing_queue.qsize()} задач в очереди...")
        await indexing_queue.join()

    if analysis_queue.qsize() > 0:
        logger.info(f"Ожидание завершения {analysis_queue.qsize()} задач анализа...")
        await analysis_queue.join()

    if search_queue.qsize() > 0:
        logger.info(f"Ожидание завершения {search_queue.qsize()} поисковых запросов...")
        await search_queue.join()

    if llm_queue.qsize() > 0:
        logger.info(f"Ожидание завершения {llm_queue.qsize()} LLM запросов...")
        await llm_queue.join()

    # Завершаем executor
    executor.shutdown(wait=True)

    # Очищаем ресурсы ререйтера если он был инициализирован
    if reranker:
        await loop.run_in_executor(executor, reranker.cleanup)
        cleanup_gpu_memory()

    # Очищаем ресурсы QdrantManager
    if qdrant_manager:
        await loop.run_in_executor(executor, qdrant_manager.cleanup)

    # Закрываем Redis соединение
    await redis_client.close()

    logger.info("Сервис остановлен")


app = FastAPI(lifespan=lifespan)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешаем все источники для тестирования
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API endpoints
@app.get("/system/status")
async def get_system_status():
    """Получает текущий статус системы"""
    try:
        status = {
            "active_indexing_tasks": active_indexing_tasks,
            "embeddings_loaded": False,
            "reranker_loaded": reranker is not None,
            "memory_info": {},
            "queues": {
                "indexing": indexing_queue.qsize() if indexing_queue else 0,
                "search": search_queue.qsize(),
                "llm": llm_queue.qsize()
            }
        }

        # Проверяем загружены ли эмбеддинги
        if qdrant_manager and hasattr(qdrant_manager, '_embeddings_initialized'):
            status["embeddings_loaded"] = qdrant_manager._embeddings_initialized

        # Получаем информацию о памяти
        if torch.cuda.is_available():
            status["memory_info"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024 ** 3:.2f} GB"
            }

        return {"status": "success", "system": status}

    except Exception as e:
        logger.error(f"Ошибка при получении статуса системы: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/status")
async def get_queue_status():
    """Получает статус всех очередей"""
    try:
        return {
            "status": "success",
            "queues": {
                "indexing": {
                    "size": indexing_queue.qsize() if indexing_queue else 0,
                    "active_tasks": active_indexing_tasks,
                    "semaphore_available": indexing_semaphore._value if indexing_semaphore else 0
                },
                "search": {
                    "size": search_queue.qsize(),
                    "semaphore_available": search_semaphore._value,
                    "max_concurrent": 3
                },
                "llm": {
                    "size": llm_queue.qsize(),
                    "semaphore_available": llm_semaphore._value,
                    "max_concurrent": 1
                }
            }
        }

    except Exception as e:
        logger.error(f"Ошибка при получении статуса очередей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queue/analysis/status")
async def get_analysis_queue_status():
    """Получает статус очереди анализа"""
    try:
        queue_size = analysis_queue.qsize() if analysis_queue else 0

        # Получаем список задач в очереди
        queue_tasks = []
        if queue_size > 0:
            # Примечание: это приблизительная информация, так как очередь может измениться
            temp_list = []
            for _ in range(queue_size):
                try:
                    item = analysis_queue.get_nowait()
                    temp_list.append(item)
                    queue_tasks.append({
                        "task_id": item.task_id,
                        "application_id": item.application_id
                    })
                except asyncio.QueueEmpty:
                    break

            # Возвращаем элементы обратно в очередь
            for item in temp_list:
                await analysis_queue.put(item)

        return {
            "status": "success",
            "queue": {
                "size": queue_size,
                "active_task": active_analysis_tasks > 0,
                "tasks": queue_tasks
            }
        }

    except Exception as e:
        logger.error(f"Ошибка при получении статуса очереди анализа: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Асинхронно получает статус задачи из Redis"""
    task_data = await redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(task_data)


@app.post("/index")
async def index_document(request: IndexDocumentRequest):
    """Запускает асинхронную индексацию документа"""
    # Добавляем задачу в очередь
    await indexing_queue.put(request)

    # Обновляем статус задачи
    await update_task_status(request.task_id, "QUEUED", 0, "queue",
                             f"Задача добавлена в очередь. Позиция: {indexing_queue.qsize()}")

    return {
        "status": "queued",
        "task_id": request.task_id,
        "queue_position": indexing_queue.qsize()
    }


@app.post("/search")
async def search(request: SearchRequest):
    """Асинхронно выполняет семантический поиск с контролем нагрузки"""
    # Создаем уникальный ID для задачи
    task_id = str(uuid.uuid4())

    # Создаем Future для результата
    future = asyncio.Future()

    # Добавляем задачу в очередь
    await search_queue.put({
        'task_id': task_id,
        'request': request,
        'future': future
    })

    # Логируем размер очереди
    queue_size = search_queue.qsize()
    if queue_size > 0:
        logger.info(f"Поисковый запрос добавлен в очередь. Позиция: {queue_size}")

    try:
        # Ждем результат
        result = await future
        return result
    except Exception as e:
        logger.exception(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
async def analyze_application(request: AnalyzeApplicationRequest):
    """Добавляет задачу анализа в очередь"""
    # Добавляем задачу в очередь
    await analysis_queue.put(request)

    # Обновляем статус задачи
    queue_position = analysis_queue.qsize()
    await update_task_status(request.task_id, "QUEUED", 0, "queue",
                             f"Задача добавлена в очередь анализа. Позиция: {queue_position}")

    return {
        "status": "queued",
        "task_id": request.task_id,
        "queue_position": queue_position
    }


@app.post("/llm/process")
async def process_llm_query(request: ProcessQueryRequest):
    """Асинхронно обрабатывает запрос через LLM с контролем нагрузки и возвратом токенов"""
    # Создаем уникальный ID для задачи
    task_id = str(uuid.uuid4())

    # Создаем Future для результата
    future = asyncio.Future()

    # Добавляем задачу в очередь
    await llm_queue.put({
        'task_id': task_id,
        'request': request,
        'future': future
    })

    # Логируем размер очереди
    queue_size = llm_queue.qsize()
    if queue_size > 0:
        logger.info(f"LLM запрос добавлен в очередь. Позиция: {queue_size}")

    try:
        # Ждем результат
        result = await future

        # Теперь результат содержит информацию о токенах
        # Формируем финальный ответ
        return {
            "status": result.get("status", "success"),
            "response": result.get("response", ""),
            "tokens": result.get("tokens", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }),
            "model": result.get("model", request.model_name)
        }

    except Exception as e:
        logger.exception(f"Ошибка LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/models")
async def get_llm_models():
    """Асинхронно получает список доступных LLM моделей"""
    try:
        loop = asyncio.get_event_loop()

        def _get_models():
            llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")
            return llm_provider.get_available_models()

        models = await loop.run_in_executor(executor, _get_models)
        return {"status": "success", "models": models}

    except Exception as e:
        logger.exception(f"Ошибка получения моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/applications/{application_id}/stats")
async def get_application_stats(application_id: str):
    """Асинхронно получает статистику по заявке"""
    try:
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(
            executor,
            qdrant_manager.get_stats,
            application_id
        )
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.exception(f"Ошибка получения статистики: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/applications/{application_id}/chunks")
async def get_application_chunks(application_id: str, limit: int = 500, offset: int = 0):
    """
    Асинхронно получает чанки заявки с поддержкой пагинации.

    Args:
        application_id: ID заявки
        limit: Максимальное количество чанков (по умолчанию 500, максимум 10000)
        offset: Смещение для пагинации (по умолчанию 0)

    Returns:
        Dict с чанками и информацией о пагинации
    """
    try:
        # Ограничиваем максимальный limit для защиты от перегрузки
        limit = min(limit, 10000)

        loop = asyncio.get_event_loop()

        def _get_chunks():
            from qdrant_client.http import models

            # Создаем фильтр для заявки
            application_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    )
                ]
            )

            # Сначала получаем общее количество чанков
            count_response = qdrant_manager.client.count(
                collection_name=qdrant_manager.collection_name,
                count_filter=application_filter,
                exact=True
            )
            total_count = count_response.count

            # Если offset больше или равен общему количеству, возвращаем пустой результат
            if offset >= total_count:
                return {
                    "chunks": [],
                    "total": total_count,
                    "offset": offset,
                    "limit": limit,
                    "has_more": False
                }

            # Для небольших датасетов или первой страницы - простой подход
            if total_count <= 1000 or (offset == 0 and limit <= 1000):
                response = qdrant_manager.client.scroll(
                    collection_name=qdrant_manager.collection_name,
                    scroll_filter=application_filter,
                    limit=total_count if total_count <= 1000 else offset + limit,
                    with_payload=True,
                    with_vectors=False
                )

                all_points = response[0]

                # Преобразуем в чанки
                all_chunks = []
                for point in all_points:
                    if hasattr(point, 'payload'):
                        chunk = {
                            "id": str(point.id),
                            "text": point.payload.get("page_content", ""),
                            "metadata": point.payload.get("metadata", {})
                        }
                        all_chunks.append(chunk)

                # Сортируем по chunk_index
                all_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

                # Применяем пагинацию
                paginated_chunks = all_chunks[offset:offset + limit]

            else:
                # Для больших датасетов с большим offset - используем scroll с итерациями
                all_chunks = []
                next_offset = None
                collected = 0

                # Собираем чанки пока не наберем нужное количество
                while collected < offset + limit:
                    batch_size = min(1000, offset + limit - collected)

                    response = qdrant_manager.client.scroll(
                        collection_name=qdrant_manager.collection_name,
                        scroll_filter=application_filter,
                        limit=batch_size,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False
                    )

                    points, next_offset = response

                    if not points:
                        break

                    # Преобразуем батч
                    for point in points:
                        if hasattr(point, 'payload'):
                            chunk = {
                                "id": str(point.id),
                                "text": point.payload.get("page_content", ""),
                                "metadata": point.payload.get("metadata", {})
                            }
                            all_chunks.append(chunk)

                    collected += len(points)

                    # Если больше нет данных
                    if next_offset is None or len(points) < batch_size:
                        break

                # Сортируем все собранные чанки
                all_chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))

                # Применяем offset и limit
                paginated_chunks = all_chunks[offset:offset + limit]

            return {
                "chunks": paginated_chunks,
                "total": total_count,
                "offset": offset,
                "limit": limit,
                "has_more": offset + limit < total_count
            }

        result = await loop.run_in_executor(executor, _get_chunks)

        # Логируем для отладки
        logger.info(f"Возвращено {len(result['chunks'])} чанков из {result['total']} "
                    f"(offset={offset}, limit={limit})")

        return {
            "status": "success",
            "chunks": result["chunks"],
            "pagination": {
                "total": result["total"],
                "offset": result["offset"],
                "limit": result["limit"],
                "has_more": result["has_more"],
                "returned": len(result["chunks"])
            }
        }

    except Exception as e:
        logger.exception(f"Ошибка получения чанков: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/applications/{application_id}")
async def delete_application_data(application_id: str):
    """Асинхронно удаляет данные заявки из векторного хранилища"""
    try:
        loop = asyncio.get_event_loop()
        deleted_count = await loop.run_in_executor(
            executor,
            qdrant_manager.delete_application,
            application_id
        )
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Удалено {deleted_count} документов"
        }
    except Exception as e:
        logger.exception(f"Ошибка удаления данных заявки: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{collection_name}/info")
async def get_collection_info(collection_name: str):
    """Получает информацию о коллекции"""
    try:
        loop = asyncio.get_event_loop()

        def _get_info():
            return {
                "exists": collection_name in [c.name for c in qdrant_manager.client.get_collections().collections],
                "vector_size": qdrant_manager.vector_size,
                "embeddings_type": "ollama",
                "model_name": qdrant_manager.model_name
            }

        info = await loop.run_in_executor(executor, _get_info)
        return {"status": "success", "info": info}

    except Exception as e:
        logger.exception(f"Ошибка получения информации о коллекции: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/cleanup")
async def manual_cleanup():
    """Ручная очистка памяти GPU - только кэши"""
    try:
        cleanup_gpu_memory()

        # Получаем информацию о памяти
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024 ** 3:.2f} GB"
            }

        return {
            "status": "success",
            "message": "GPU кэши очищены",
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Ошибка при очистке памяти: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preload-models")
async def preload_models():
    """Предварительная загрузка моделей в память (опционально)"""
    try:
        loop = asyncio.get_event_loop()

        # Загружаем эмбеддинги
        logger.info("Предзагрузка эмбеддингов...")
        await loop.run_in_executor(
            executor,
            lambda: qdrant_manager.embeddings  # Вызов property инициализирует эмбеддинги
        )

        # Загружаем ререйтер
        logger.info("Предзагрузка ререйтера...")
        await get_reranker()

        # Получаем информацию о памяти
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024 ** 3:.2f} GB"
            }

        return {
            "status": "success",
            "message": "Модели загружены в память",
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Ошибка при предзагрузке моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unload-models")
async def unload_models():
    """Выгрузка моделей из памяти для освобождения VRAM"""
    global reranker, reranker_initialized, qdrant_manager

    try:
        loop = asyncio.get_event_loop()

        # Выгружаем ререйтер
        if reranker:
            await loop.run_in_executor(executor, reranker.cleanup)
            reranker = None
            reranker_initialized = False
            logger.info("Ререйтер выгружен")

        # Выгружаем эмбеддинги из QdrantManager
        if qdrant_manager:
            await loop.run_in_executor(executor, qdrant_manager.cleanup)
            logger.info("QdrantManager очищен")

        # Очищаем память
        cleanup_gpu_memory()

        # Получаем информацию о памяти
        memory_info = {}
        if torch.cuda.is_available():
            memory_info = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB",
                "free": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024 ** 3:.2f} GB"
            }

        return {
            "status": "success",
            "message": "Модели выгружены из памяти",
            "memory_info": memory_info
        }
    except Exception as e:
        logger.error(f"Ошибка при выгрузке моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Добавьте эти эндпоинты в ваш FastAPI main.py файл после существующего эндпоинта get_task_status

@app.get("/tasks/{task_id}/status")
async def get_task_status_plural(task_id: str):
    """Асинхронно получает статус задачи из Redis (альтернативный URL с 's')"""
    # Просто вызываем существующую функцию
    return await get_task_status(task_id)


@app.get("/tasks/{task_id}/results")
async def get_task_results(task_id: str):
    """Получает результаты выполненной задачи"""
    # Сначала пробуем получить из отдельного ключа результатов
    results_data = await redis_client.get(f"task_results:{task_id}")
    if results_data:
        return json.loads(results_data)

    # Если нет результатов в отдельном ключе, проверяем основной ключ задачи
    task_data = await redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    data = json.loads(task_data)

    # НОВОЕ: Проверяем статус, но разрешаем PROGRESS для промежуточных результатов
    status = data.get("status")
    if status not in ["SUCCESS", "PROGRESS"]:
        # Для обратной совместимости возвращаем ошибку 400 для других статусов
        raise HTTPException(status_code=400, detail=f"Task not completed. Status: {status}")

    # Извлекаем результаты из данных задачи
    if "result" in data and isinstance(data["result"], dict):
        results = data["result"].get("results", [])
        return {"results": results}

    return {"results": []}


@app.get("/task/{task_id}/results")
async def get_task_results_singular(task_id: str):
    """Получает результаты выполненной задачи (альтернативный URL без 's')"""
    # Просто вызываем функцию с множественным числом
    return await get_task_results(task_id)


# Добавьте этот эндпоинт в ваш FastAPI main.py файл

@app.delete("/applications/{application_id}/documents/{document_id}")
async def delete_document_chunks(application_id: str, document_id: str):
    """Удаляет чанки конкретного документа из векторного хранилища"""
    try:
        loop = asyncio.get_event_loop()

        def _delete_chunks():
            from qdrant_client.http import models

            # Создаем фильтр для удаления чанков конкретного документа
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    ),
                    models.FieldCondition(
                        key="metadata.document_id",
                        match=models.MatchValue(value=document_id)
                    )
                ]
            )

            # Сначала получаем количество документов для удаления
            scroll_result = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name,
                scroll_filter=delete_filter,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )

            points_to_delete = [point.id for point in scroll_result[0]]
            deleted_count = len(points_to_delete)

            # Удаляем найденные точки
            if points_to_delete:
                qdrant_manager.client.delete(
                    collection_name=qdrant_manager.collection_name,
                    points_selector=models.PointIdsList(
                        points=points_to_delete
                    )
                )

                logger.info(f"Удалено {deleted_count} чанков документа {document_id} из заявки {application_id}")

            return deleted_count

        deleted_count = await loop.run_in_executor(executor, _delete_chunks)

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Удалено {deleted_count} чанков документа {document_id}"
        }

    except Exception as e:
        logger.exception(f"Ошибка удаления чанков документа: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/system/stats")
async def get_system_stats():
    """Получение статистики использования системных ресурсов"""
    try:
        loop = asyncio.get_event_loop()

        # Асинхронная функция для получения информации о GPU
        async def get_gpu_info_async():
            def _get_gpu_info():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Берем первую GPU
                        return {
                            "name": gpu.name,
                            "vram_percent": round(gpu.memoryUtil * 100, 1),
                            "vram_used_gb": round(gpu.memoryUsed / 1024, 2),
                            "vram_total_gb": round(gpu.memoryTotal / 1024, 2),
                            "temperature": gpu.temperature,
                            "utilization": round(gpu.load * 100, 1)
                        }
                except Exception as e:
                    logger.warning(f"Не удалось получить информацию о GPU: {e}")
                    return None

            return await loop.run_in_executor(executor, _get_gpu_info)

        # Асинхронная функция для получения CPU процента
        async def get_cpu_percent_async():
            def _get_cpu_percent():
                return psutil.cpu_percent(interval=1)

            return await loop.run_in_executor(executor, _get_cpu_percent)

        # Асинхронная функция для получения информации о памяти
        async def get_memory_info_async():
            def _get_memory_info():
                memory = psutil.virtual_memory()
                return {
                    "percent": round(memory.percent, 1),
                    "used_gb": round(memory.used / (1024 ** 3), 2),
                    "total_gb": round(memory.total / (1024 ** 3), 2),
                    "available_gb": round(memory.available / (1024 ** 3), 2)
                }

            return await loop.run_in_executor(executor, _get_memory_info)

        # Асинхронная функция для получения информации о дисках
        async def get_disk_info_async():
            def _get_disk_info():
                disk_usage = psutil.disk_usage('/')
                return round(disk_usage.percent, 1)

            return await loop.run_in_executor(executor, _get_disk_info)

        # Асинхронная функция для подсчета процессов
        async def get_process_count_async():
            def _get_process_count():
                return len(psutil.pids())

            return await loop.run_in_executor(executor, _get_process_count)

        # Асинхронная функция для получения информации о CPU (ядра и потоки)
        async def get_cpu_static_info_async():
            def _get_cpu_static_info():
                return {
                    "cores": psutil.cpu_count(logical=False),
                    "threads": psutil.cpu_count(logical=True)
                }

            return await loop.run_in_executor(executor, _get_cpu_static_info)

        # Запускаем все задачи параллельно с таймаутами
        tasks = [
            asyncio.create_task(asyncio.wait_for(get_cpu_percent_async(), timeout=1.5)),
            asyncio.create_task(asyncio.wait_for(get_memory_info_async(), timeout=0.5)),
            asyncio.create_task(asyncio.wait_for(get_disk_info_async(), timeout=0.5)),
            asyncio.create_task(asyncio.wait_for(get_process_count_async(), timeout=0.5)),
            asyncio.create_task(asyncio.wait_for(get_cpu_static_info_async(), timeout=0.5)),
            asyncio.create_task(asyncio.wait_for(get_gpu_info_async(), timeout=1.0))
        ]

        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем результаты
        cpu_percent = results[0] if not isinstance(results[0], Exception) else 0
        memory_info = results[1] if not isinstance(results[1], Exception) else {
            "percent": 0, "used_gb": 0, "total_gb": 0, "available_gb": 0
        }
        disk_percent = results[2] if not isinstance(results[2], Exception) else 0
        process_count = results[3] if not isinstance(results[3], Exception) else 0
        cpu_static_info = results[4] if not isinstance(results[4], Exception) else {
            "cores": 0, "threads": 0
        }
        gpu_info = results[5] if not isinstance(results[5], Exception) else None

        # Логируем ошибки, если они были
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                if isinstance(result, asyncio.TimeoutError):
                    logger.warning(f"Таймаут при выполнении задачи {i}")
                else:
                    logger.warning(f"Ошибка при выполнении задачи {i}: {result}")

        # Формируем ответ
        response = {
            "cpu": {
                "percent": round(cpu_percent, 1),
                "cores": cpu_static_info.get("cores", 0),
                "threads": cpu_static_info.get("threads", 0)
            },
            "memory": memory_info,
            "gpu": gpu_info or {
                "name": "Не обнаружено",
                "vram_percent": 0,
                "vram_used_gb": 0,
                "vram_total_gb": 0,
                "temperature": None,
                "utilization": 0
            },
            "system": {
                "process_count": process_count,
                "disk_percent": disk_percent,
                "active_indexing_tasks": active_indexing_tasks,
                "queues": {
                    "indexing": indexing_queue.qsize() if indexing_queue else 0,
                    "search": search_queue.qsize(),
                    "llm": llm_queue.qsize()
                }
            }
        }

        return response

    except Exception as e:
        logger.error(f"Критическая ошибка при получении системной статистики: {e}")
        # Возвращаем минимальную статистику вместо ошибки
        return {
            "error": str(e),
            "cpu": {"percent": 0, "cores": 0, "threads": 0},
            "memory": {"percent": 0, "used_gb": 0, "total_gb": 0, "available_gb": 0},
            "gpu": {"name": "Ошибка", "vram_percent": 0, "vram_used_gb": 0,
                    "vram_total_gb": 0, "temperature": None, "utilization": 0},
            "system": {"process_count": 0, "disk_percent": 0,
                       "active_indexing_tasks": 0,
                       "indexing_queue_size": 0}
        }


@app.delete("/applications/{application_id}/files/{file_id}/chunks")
async def delete_file_chunks(application_id: str, file_id: str):
    """Удаляет чанки по file_id из метаданных"""
    try:
        loop = asyncio.get_event_loop()

        def _delete_chunks():
            from qdrant_client.http import models

            # Создаем фильтр для удаления чанков конкретного файла
            delete_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    ),
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )

            # Сначала получаем количество документов для удаления
            scroll_result = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name,
                scroll_filter=delete_filter,
                limit=1000,
                with_payload=False,
                with_vectors=False
            )

            points_to_delete = [point.id for point in scroll_result[0]]
            deleted_count = len(points_to_delete)

            # Удаляем найденные точки
            if points_to_delete:
                qdrant_manager.client.delete(
                    collection_name=qdrant_manager.collection_name,
                    points_selector=models.PointIdsList(
                        points=points_to_delete
                    )
                )

                logger.info(f"Удалено {deleted_count} чанков файла {file_id} из заявки {application_id}")

            return deleted_count

        deleted_count = await loop.run_in_executor(executor, _delete_chunks)

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "message": f"Удалено {deleted_count} чанков файла {file_id}"
        }

    except Exception as e:
        logger.exception(f"Ошибка удаления чанков файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/applications/{application_id}/files/{file_id}/stats")
async def get_file_stats(application_id: str, file_id: str):
    """Получает статистику по конкретному файлу"""
    try:
        loop = asyncio.get_event_loop()

        def _get_stats():
            from qdrant_client.http import models

            # Создаем фильтр для поиска чанков файла
            file_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.application_id",
                        match=models.MatchValue(value=application_id)
                    ),
                    models.FieldCondition(
                        key="metadata.file_id",
                        match=models.MatchValue(value=file_id)
                    )
                ]
            )

            # Получаем чанки файла
            scroll_result = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name,
                scroll_filter=file_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            chunks_count = len(scroll_result[0])

            # Собираем статистику по типам контента
            content_types = {}
            for point in scroll_result[0]:
                if hasattr(point, 'payload') and 'metadata' in point.payload:
                    content_type = point.payload['metadata'].get('content_type', 'unknown')
                    content_types[content_type] = content_types.get(content_type, 0) + 1

            return {
                "chunks_count": chunks_count,
                "content_types": content_types,
                "file_id": file_id,
                "application_id": application_id
            }

        stats = await loop.run_in_executor(executor, _get_stats)

        return {
            "status": "success",
            "chunks_count": stats["chunks_count"],
            "content_types": stats["content_types"]
        }

    except Exception as e:
        logger.exception(f"Ошибка получения статистики файла: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/applications/{application_id}/files")
async def get_application_files(application_id: str):
    """Получает список всех файлов в заявке на основе метаданных"""
    try:
        loop = asyncio.get_event_loop()

        def _get_files():
            from qdrant_client.http import models

            # Получаем все чанки заявки
            scroll_result = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.application_id",
                            match=models.MatchValue(value=application_id)
                        )
                    ]
                ),
                limit=10000,
                with_payload=True,
                with_vectors=False
            )

            # Собираем уникальные файлы
            files_info = {}
            for point in scroll_result[0]:
                if hasattr(point, 'payload') and 'metadata' in point.payload:
                    metadata = point.payload['metadata']
                    file_id = metadata.get('file_id')

                    if file_id and file_id not in files_info:
                        files_info[file_id] = {
                            "file_id": file_id,
                            "document_id": metadata.get('document_id'),
                            "document_name": metadata.get('document_name', 'Unknown'),
                            "chunks_count": 0
                        }

                    if file_id:
                        files_info[file_id]["chunks_count"] += 1

            return list(files_info.values())

        files = await loop.run_in_executor(executor, _get_files)

        return {
            "status": "success",
            "files": files,
            "total_files": len(files)
        }

    except Exception as e:
        logger.exception(f"Ошибка получения списка файлов: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)