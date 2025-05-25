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

# Импорты из ppee_analyzer
from ppee_analyzer.vector_store import QdrantManager, BGEReranker, OllamaEmbeddings
from ppee_analyzer.semantic_chunker import SemanticChunker
from ppee_analyzer.checklist import ChecklistAnalyzer
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Глобальные переменные
redis_client = None
qdrant_manager = None
reranker = None
executor = ThreadPoolExecutor(max_workers=10)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, qdrant_manager, reranker

    # Инициализация Redis с асинхронным клиентом
    redis_client = await redis.from_url("redis://localhost:6379", decode_responses=True)

    # Инициализация QdrantManager в отдельном потоке
    loop = asyncio.get_event_loop()
    qdrant_manager = await loop.run_in_executor(
        executor,
        lambda: QdrantManager(
            collection_name="ppee_applications",
            host="localhost",
            port=6333,
            embeddings_type="ollama",
            model_name="bge-m3",
            ollama_url="http://localhost:11434"
        )
    )

    # Инициализация ререйтера (опционально)
    try:
        reranker = await loop.run_in_executor(
            executor,
            lambda: BGEReranker(
                model_name="BAAI/bge-reranker-v2-m3",
                device="cuda",
                min_vram_mb=500
            )
        )
    except Exception as e:
        logger.warning(f"Не удалось инициализировать ререйтер: {e}")
        reranker = None

    yield

    # Shutdown
    executor.shutdown(wait=True)
    if reranker:
        await loop.run_in_executor(executor, reranker.cleanup)
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

    return await loop.run_in_executor(executor, _process)


# API endpoints
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
    # Создаем задачу
    asyncio.create_task(index_document_task(request))
    return {"status": "started", "task_id": request.task_id}


async def index_document_task(request: IndexDocumentRequest):
    """Асинхронная задача индексации"""
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
    """Векторный поиск (синхронная функция для executor)"""
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

    # Ререйтинг
    if use_reranker and reranker and results:
        results = reranker.rerank(query, results, top_k=limit, text_key="text")

    return results[:limit]


def hybrid_search(application_id: str, query: str, limit: int,
                  vector_weight: float, text_weight: float, use_reranker: bool) -> List[Dict]:
    """Гибридный поиск (синхронная функция для executor)"""
    # Векторный поиск
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
    if use_reranker and reranker and combined:
        combined = reranker.rerank(query, combined, top_k=limit, text_key="text")

    return combined[:limit]


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
    """Запускает асинхронный анализ заявки"""
    asyncio.create_task(analyze_application_task(request))
    return {"status": "started", "task_id": request.task_id}


async def analyze_application_task(request: AnalyzeApplicationRequest):
    """Асинхронная задача анализа"""
    try:
        await update_task_status(request.task_id, "PROGRESS", 10, "prepare", "Инициализация анализа...")

        loop = asyncio.get_event_loop()
        analyzer = ChecklistAnalyzer(qdrant_manager)
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
                    from langchain_core.documents import Document
                    doc = Document(
                        page_content=res.get('text', ''),
                        metadata=res.get('metadata', {})
                    )
                    search_documents.append(doc)

                # Обработка через LLM
                if search_documents:
                    # Форматируем контекст
                    context = "\n\n".join([doc.page_content for doc in search_documents])

                    # Вызываем LLM асинхронно
                    llm_response = await process_llm_query_async(
                        item['llm_model'],
                        item['llm_prompt_template'],
                        context,
                        {
                            'temperature': item.get('llm_temperature', 0.1),
                            'max_tokens': item.get('llm_max_tokens', 1000),
                            'search_query': query
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
                            'prompt_template': item['llm_prompt_template'],
                            'query': query,
                            'context': context,
                            'model': item['llm_model'],
                            'temperature': item.get('llm_temperature', 0.1),
                            'max_tokens': item.get('llm_max_tokens', 1000),
                            'response': llm_response,
                            'search_method': search_method
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
                            'search_method': search_method
                        }
                    })

            except Exception as e:
                logger.error(f"Ошибка при обработке параметра {item['id']}: {e}")
                error_count += 1

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
        await update_task_status(request.task_id, "FAILURE", 0, "error", str(e))


# Добавляем вспомогательные функции для извлечения значений
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
        from app.adapters.llm_adapter import OllamaLLMProvider
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
            from app.adapters.llm_adapter import OllamaLLMProvider
            llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")
            return llm_provider.get_available_models()

        models = await loop.run_in_executor(executor, _get_models)
        return {"status": "success", "models": models}

    except Exception as e:
        logger.exception(f"Ошибка получения моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Добавляем в существующий main.py

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

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)