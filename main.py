from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uuid
import redis
import json
import logging
import os
from contextlib import asynccontextmanager

# Импорты из ppee_analyzer
from ppee_analyzer.vector_store import QdrantManager, BGEReranker, OllamaEmbeddings
from ppee_analyzer.semantic_chunker import SemanticChunker
from ppee_analyzer.checklist import ChecklistAnalyzer
from langchain_core.documents import Document

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis клиент
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Глобальные переменные для менеджеров
qdrant_manager = None
reranker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global qdrant_manager, reranker

    # Инициализация QdrantManager
    qdrant_manager = QdrantManager(
        collection_name="ppee_applications",
        host="localhost",
        port=6333,
        embeddings_type="ollama",
        model_name="bge-m3",
        ollama_url="http://localhost:11434"
    )

    # Инициализация ререйтера (опционально)
    try:
        reranker = BGEReranker(
            model_name="BAAI/bge-reranker-v2-m3",
            device="cuda",
            min_vram_mb=500
        )
    except Exception as e:
        logger.warning(f"Не удалось инициализировать ререйтер: {e}")
        reranker = None

    yield

    # Shutdown
    if reranker:
        reranker.cleanup()


app = FastAPI(lifespan=lifespan)


# Pydantic модели для запросов
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


# Вспомогательные функции
def update_task_status(task_id: str, status: str, progress: int = 0,
                       stage: str = "", message: str = "", result: Any = None):
    """Обновляет статус задачи в Redis"""
    task_data = {
        "status": status,
        "progress": progress,
        "stage": stage,
        "message": message
    }
    if result is not None:
        task_data["result"] = result

    redis_client.setex(f"task:{task_id}", 3600, json.dumps(task_data))
    logger.info(f"Task {task_id}: {status} - {progress}% [{stage}] {message}")


def process_document_with_semantic_chunker(document_path: str, application_id: str) -> List[Document]:
    """Обрабатывает документ с использованием семантического чанкера"""
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


# API endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Получает статус задачи из Redis"""
    task_data = redis_client.get(f"task:{task_id}")
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found")

    return json.loads(task_data)


@app.post("/index")
async def index_document(request: IndexDocumentRequest, background_tasks: BackgroundTasks):
    """Индексирует документ в Qdrant"""
    # Запускаем в фоне
    background_tasks.add_task(index_document_task, request)
    return {"status": "started", "task_id": request.task_id}


async def index_document_task(request: IndexDocumentRequest):
    """Фоновая задача индексации"""
    try:
        update_task_status(request.task_id, "PROGRESS", 5, "prepare", "Подготовка к индексации...")

        # Проверяем файл
        if not os.path.exists(request.document_path):
            raise FileNotFoundError(f"Файл не найден: {request.document_path}")

        update_task_status(request.task_id, "PROGRESS", 20, "convert", "Обработка документа...")

        # Обрабатываем документ
        chunks = process_document_with_semantic_chunker(request.document_path, request.application_id)

        update_task_status(request.task_id, "PROGRESS", 50, "index", f"Индексация {len(chunks)} фрагментов...")

        # Удаляем старые данные если нужно
        if request.delete_existing:
            qdrant_manager.delete_application(request.application_id)

        # Индексируем
        total_chunks = len(chunks)
        batch_size = 20

        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch = chunks[i:end_idx]
            qdrant_manager.add_documents(batch)

            progress = 50 + int(45 * (end_idx / total_chunks))
            update_task_status(request.task_id, "PROGRESS", progress, "index",
                               f"Индексация: {end_idx}/{total_chunks}...")

        # Успешное завершение
        result = {
            "application_id": request.application_id,
            "document_path": request.document_path,
            "total_chunks": total_chunks,
            "status": "success"
        }

        update_task_status(request.task_id, "SUCCESS", 100, "complete",
                           "Индексация завершена", result)

    except Exception as e:
        logger.exception(f"Ошибка индексации: {e}")
        update_task_status(request.task_id, "FAILURE", 0, "error", str(e))


@app.post("/search")
async def search(request: SearchRequest):
    """Выполняет семантический поиск"""
    try:
        # Выполняем поиск
        if request.use_smart_search:
            # Умный поиск - выбор метода в зависимости от длины запроса
            if len(request.query) < request.hybrid_threshold:
                # Гибридный поиск
                results = hybrid_search(
                    request.application_id,
                    request.query,
                    request.limit,
                    request.vector_weight,
                    request.text_weight,
                    request.use_reranker
                )
            else:
                # Векторный поиск
                results = vector_search(
                    request.application_id,
                    request.query,
                    request.limit,
                    request.use_reranker,
                    request.rerank_limit
                )
        else:
            # Обычный векторный поиск
            results = vector_search(
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
    """Векторный поиск"""
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
    """Гибридный поиск"""
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
async def analyze_application(request: AnalyzeApplicationRequest, background_tasks: BackgroundTasks):
    """Анализирует заявку по чек-листам"""
    background_tasks.add_task(analyze_application_task, request)
    return {"status": "started", "task_id": request.task_id}


async def analyze_application_task(request: AnalyzeApplicationRequest):
    """Фоновая задача анализа"""
    try:
        update_task_status(request.task_id, "PROGRESS", 10, "prepare", "Инициализация анализа...")

        analyzer = ChecklistAnalyzer(qdrant_manager)
        processed_count = 0
        error_count = 0
        total_items = len(request.checklist_items)

        results = []

        for i, item in enumerate(request.checklist_items):
            try:
                progress = 15 + int(75 * (i / total_items))
                update_task_status(request.task_id, "PROGRESS", progress, "analyze",
                                   f"Анализ параметра {i + 1}/{total_items}: {item['name']}")

                # Поиск
                search_results = qdrant_manager.search(
                    query=item['search_query'],
                    filter_dict={"application_id": request.application_id},
                    k=item.get('search_limit', 3)
                )

                # Обработка через LLM
                if search_results:
                    from app.adapters.llm_adapter import OllamaLLMProvider
                    llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")

                    # Форматируем контекст
                    context = "\n\n".join([doc.page_content for doc in search_results])

                    # Вызываем LLM
                    llm_response = llm_provider.process_query(
                        model_name=item['llm_model'],
                        prompt=item['llm_prompt_template'],
                        context=context,
                        parameters=request.llm_params,
                        query=item['search_query']
                    )

                    results.append({
                        "parameter_id": item['id'],
                        "value": llm_response,
                        "confidence": 0.8,
                        "search_results": [{"text": doc.page_content, "metadata": doc.metadata}
                                           for doc in search_results]
                    })
                    processed_count += 1
                else:
                    results.append({
                        "parameter_id": item['id'],
                        "value": "Информация не найдена",
                        "confidence": 0.0,
                        "search_results": []
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

        update_task_status(request.task_id, "SUCCESS", 100, "complete",
                           "Анализ завершен", result)

    except Exception as e:
        logger.exception(f"Ошибка анализа: {e}")
        update_task_status(request.task_id, "FAILURE", 0, "error", str(e))


@app.post("/llm/process")
async def process_llm_query(request: ProcessQueryRequest):
    """Обрабатывает запрос через LLM"""
    try:
        from app.adapters.llm_adapter import OllamaLLMProvider
        llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")

        response = llm_provider.process_query(
            model_name=request.model_name,
            prompt=request.prompt,
            context=request.context,
            parameters=request.parameters,
            query=request.query
        )

        return {"status": "success", "response": response}

    except Exception as e:
        logger.exception(f"Ошибка LLM: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm/models")
async def get_llm_models():
    """Получает список доступных LLM моделей"""
    try:
        from app.adapters.llm_adapter import OllamaLLMProvider
        llm_provider = OllamaLLMProvider(base_url="http://localhost:11434")
        models = llm_provider.get_available_models()
        return {"status": "success", "models": models}
    except Exception as e:
        logger.exception(f"Ошибка получения моделей: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)