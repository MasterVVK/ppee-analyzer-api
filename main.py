from fastapi import FastAPI, HTTPException, BackgroundTasks
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

# Импорты из ppee_analyzer
from ppee_analyzer.vector_store import QdrantManager, BGEReranker, OllamaEmbeddings
from ppee_analyzer.semantic_chunker import SemanticChunker
#from ppee_analyzer.checklist import ChecklistAnalyzer
from langchain_core.documents import Document

# Импорты из локальных адаптеров
from app.adapters.llm_adapter import OllamaLLMProvider

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
redis_client = None
qdrant_manager = None
reranker = None
executor = ThreadPoolExecutor(max_workers=10)
reranker_semaphore = None
reranker_initialized = False  # Флаг инициализации ререйтера
active_indexing_tasks = 0  # Счетчик активных задач индексации
indexing_lock = None  # Блокировка для счетчика
indexing_queue = None  # Очередь для задач индексации
indexing_semaphore = None  # Семафор для ограничения одной индексации

# Настройки очистки памяти
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10s")  # Время хранения модели Ollama в памяти


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, qdrant_manager, reranker_semaphore, indexing_lock, indexing_queue, indexing_semaphore

    # Инициализация семафора для ререйтера
    reranker_semaphore = asyncio.Semaphore(1)

    # Инициализация блокировки для счетчика индексаций
    indexing_lock = asyncio.Lock()

    # Инициализация очереди для задач индексации
    indexing_queue = asyncio.Queue()

    # Семафор для ограничения одной индексации за раз
    indexing_semaphore = asyncio.Semaphore(1)

    # Инициализация Redis с асинхронным клиентом
    redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)

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

    # НЕ инициализируем ререйтер при старте
    logger.info("Сервис запущен. Модели будут загружены при первом использовании.")

    # Запускаем периодическую очистку памяти
    cleanup_task = asyncio.create_task(periodic_cleanup())

    # Запускаем обработчик очереди индексации
    indexing_worker_task = asyncio.create_task(indexing_queue_worker())

    yield

    # Shutdown
    cleanup_task.cancel()
    indexing_worker_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass
    try:
        await indexing_worker_task
    except asyncio.CancelledError:
        pass

    executor.shutdown(wait=True)
    if reranker:
        await loop.run_in_executor(executor, reranker.cleanup)
        cleanup_gpu_memory()
    await redis_client.close()


app = FastAPI(lifespan=lifespan)


# Pydantic модели
class IndexDocumentRequest(BaseModel):
    task_id: str
    application_id: str
    document_path: str
    delete_existing: bool = False


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


class AnalyzeApplicationRequest(BaseModel):
    task_id: str
    application_id: str
    checklist_items: List[Dict[str, Any]]
    llm_params: Dict[str, Any]


class ProcessQueryRequest(BaseModel):
    model_name: str
    prompt: str
    context: str
    parameters: Dict[str, Any]
    query: Optional[str] = None


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


# Функция для ленивой инициализации ререйтера
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


# Асинхронные вспомогательные функции
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

    await redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_data))
    logger.info(f"Task {task_id}: {status} - {progress}% [{stage}] {message}")


async def process_document_with_semantic_chunker(document_path: str, application_id: str) -> List[Document]:
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

            # Преобразуем в Document объекты
            documents = []
            document_id = f"doc_{os.path.basename(document_path).replace(' ', '_').replace('.', '_')}"
            document_name = os.path.basename(document_path)

            for i, chunk in enumerate(grouped_chunks):
                metadata = {
                    "application_id": application_id,
                    "document_id": document_id,
                    "document_name": document_name,
                    "content_type": chunk.get("type", "unknown"),
                    "chunk_index": i,
                    "section": chunk.get("heading", "Не определено"),
                }

                if chunk.get("page"):
                    metadata["page_number"] = chunk.get("page")

                documents.append(Document(
                    page_content=chunk.get("content", ""),
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

                # Преобразуем в Document объекты
                documents = []
                document_id = f"doc_{os.path.basename(document_path).replace(' ', '_').replace('.', '_')}"
                document_name = os.path.basename(document_path)

                for i, chunk in enumerate(grouped_chunks):
                    metadata = {
                        "application_id": application_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "content_type": chunk.get("type", "unknown"),
                        "chunk_index": i,
                        "section": chunk.get("heading", "Не определено"),
                    }

                    if chunk.get("page"):
                        metadata["page_number"] = chunk.get("page")

                    documents.append(Document(
                        page_content=chunk.get("content", ""),
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
            "indexing_queue_size": indexing_queue.qsize() if indexing_queue else 0
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
    """Получает статус очереди индексации"""
    try:
        queue_size = indexing_queue.qsize() if indexing_queue else 0

        # Получаем список задач в очереди
        queue_tasks = []
        if queue_size > 0:
            # Примечание: это приблизительная информация, так как очередь может измениться
            temp_list = []
            for _ in range(queue_size):
                try:
                    item = indexing_queue.get_nowait()
                    temp_list.append(item)
                    queue_tasks.append({
                        "task_id": item.task_id,
                        "application_id": item.application_id,
                        "document_path": os.path.basename(item.document_path)
                    })
                except asyncio.QueueEmpty:
                    break

            # Возвращаем элементы обратно в очередь
            for item in temp_list:
                await indexing_queue.put(item)

        return {
            "status": "success",
            "queue": {
                "size": queue_size,
                "active_task": active_indexing_tasks > 0,
                "tasks": queue_tasks
            }
        }

    except Exception as e:
        logger.error(f"Ошибка при получении статуса очереди: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


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

        await update_task_status(request.task_id, "PROGRESS", 20, "convert", "Обработка документа...")

        # Обрабатываем документ
        chunks = await process_document_with_semantic_chunker(request.document_path, request.application_id)

        await update_task_status(request.task_id, "PROGRESS", 50, "index", f"Индексация {len(chunks)} фрагментов...")

        # Удаляем старые данные если нужно
        if request.delete_existing:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, qdrant_manager.delete_application, request.application_id)

        # Индексируем пакетами асинхронно
        total_chunks = len(chunks)
        batch_size = 20

        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]

            # Добавляем документы в отдельном потоке
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, qdrant_manager.add_documents, batch)

            progress = 50 + int(45 * (end_idx / total_chunks))
            await update_task_status(request.task_id, "PROGRESS", progress, "index",
                                     f"Индексация: {end_idx}/{total_chunks}...")

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

        # ИЗМЕНЕНИЕ: Убираем агрессивную очистку, оставляем только легкую очистку GPU кэшей
        cleanup_gpu_memory()


async def index_document_task(request: IndexDocumentRequest):
    """Устаревшая функция для совместимости"""
    logger.warning("Использование устаревшей функции index_document_task")
    await index_document_task_worker(request)


@app.post("/search")
async def search(request: SearchRequest):
    """Асинхронно выполняет семантический поиск"""
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

        return {"status": "success", "results": results}

    except Exception as e:
        logger.exception(f"Ошибка поиска: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
                    # ВАЖНО: Освобождаем только кэши после ререйтинга
                    cleanup_gpu_memory()

        return results[:limit]
    except Exception as e:
        # При ошибке тоже освобождаем кэши
        cleanup_gpu_memory()
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
                    # ВАЖНО: Освобождаем только кэши после ререйтинга
                    cleanup_gpu_memory()

        return combined[:limit]
    except Exception as e:
        # При ошибке тоже освобождаем кэши
        cleanup_gpu_memory()
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


@app.post("/analyze")
async def analyze_application(request: AnalyzeApplicationRequest):
#    Запускает асинхронный анализ заявки
    asyncio.create_task(analyze_application_task(request))
    return {"status": "started", "task_id": request.task_id}


async def analyze_application_task(request: AnalyzeApplicationRequest):
    """Асинхронная задача анализа с умным поиском"""
    try:
        await update_task_status(request.task_id, "PROGRESS", 10, "prepare", "Инициализация анализа...")

        loop = asyncio.get_event_loop()
        #analyzer = ChecklistAnalyzer(qdrant_manager)
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

                # Обработка через LLM
                if search_documents:
                    # Форматируем контекст
                    context = "\n\n".join([doc.page_content for doc in search_documents])

                    # ВАЖНО: Формируем полный промпт здесь
                    full_prompt = item['llm_prompt_template'].replace(
                        "{query}", query
                    ).replace(
                        "{context}", context
                    )

                    # Вызываем LLM асинхронно с полным промптом
                    llm_response = await process_llm_query_async(
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

                    # Извлекаем значение
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
                            'prompt': full_prompt,  # Сохраняем ПОЛНЫЙ промпт
                            'prompt_template': item['llm_prompt_template'],  # И шаблон для справки
                            'model': item['llm_model'],
                            'temperature': item.get('llm_temperature', 0.1),
                            'max_tokens': item.get('llm_max_tokens', 1000),
                            'context_length': item.get('context_length', 8192),
                            'response': llm_response,  # Полный ответ LLM
                            'search_query': query,
                            'search_method': search_method,
                            'context': context  # Можно сохранить и контекст для отладки
                        }
                    })
                    processed_count += 1
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
                            'search_query': query
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
                        'model': item.get('llm_model', 'unknown')
                    }
                })

        # Освобождаем кэши после завершения анализа
        cleanup_gpu_memory()

        # Завершение
        result = {
            "status": "success",
            "processed": processed_count,
            "errors": error_count,
            "total": total_items,
            "results": results
        }

        await update_task_status(request.task_id, "SUCCESS", 100, "complete",
                                 "Анализ завершен", result)

    except Exception as e:
        logger.exception(f"Ошибка анализа: {e}")
        cleanup_gpu_memory()  # Освобождаем кэши при ошибке
        await update_task_status(request.task_id, "FAILURE", 0, "error", str(e))


# Вспомогательные функции для анализа
def extract_value_from_response(response: str, query: str) -> str:
    """Извлекает значение из ответа LLM"""
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


def calculate_confidence(response: str) -> float:
    """Рассчитывает уверенность в ответе"""
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


async def process_llm_query_async(model_name: str, prompt: str, context: str,
                                  parameters: Dict[str, Any], query: Optional[str] = None):
    """Асинхронная обработка запроса через LLM"""
    loop = asyncio.get_event_loop()

    def _process():
        llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")

        return llm_provider.process_query(
            model_name=model_name,
            prompt=prompt,
            context=context,
            parameters=parameters,
            query=query
        )

    return await loop.run_in_executor(executor, _process)


@app.post("/llm/process")
async def process_llm_query(request: ProcessQueryRequest):
    """Асинхронно обрабатывает запрос через LLM"""
    try:
        response = await process_llm_query_async(
            request.model_name,
            request.prompt,
            request.context,
            request.parameters,
            request.query
        )

        return {"status": "success", "response": response}

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
async def get_application_chunks(application_id: str, limit: int = 500):
    """Асинхронно получает чанки заявки"""
    try:
        loop = asyncio.get_event_loop()

        def _get_chunks():
            from qdrant_client.http import models

            response = qdrant_manager.client.scroll(
                collection_name=qdrant_manager.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.application_id",
                            match=models.MatchValue(value=application_id)
                        )
                    ]
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            # Преобразуем результаты
            chunks = []
            for point in response[0]:
                if hasattr(point, 'payload'):
                    chunk = {
                        "id": point.id,
                        "text": point.payload.get("page_content", ""),
                        "metadata": point.payload.get("metadata", {})
                    }
                    chunks.append(chunk)

            # Сортируем по chunk_index
            chunks.sort(key=lambda x: x["metadata"].get("chunk_index", 0))
            return chunks

        chunks = await loop.run_in_executor(executor, _get_chunks)
        return {"status": "success", "chunks": chunks}

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
    task_data = await redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    data = json.loads(task_data)

    # Проверяем, что задача завершена успешно
    if data.get("status") != "SUCCESS":
        raise HTTPException(status_code=400, detail=f"Task not completed. Status: {data.get('status')}")

    # Возвращаем результаты
    result = data.get("result", {})
    return {
        "status": "success",
        "task_id": task_id,
        "results": result.get("results", []) if isinstance(result, dict) else []
    }


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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)