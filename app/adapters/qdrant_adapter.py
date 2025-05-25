import os
import logging
from typing import List, Dict, Any, Optional

from ppee_analyzer.vector_store import QdrantManager, BGEReranker
from langchain_core.documents import Document
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantAdapter:
    """Адаптер для работы с Qdrant через ppee_analyzer"""

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6333,
                 collection_name: str = "ppee_applications",
                 embeddings_type: str = "ollama",
                 model_name: str = "bge-m3",
                 device: str = "cuda",
                 ollama_url: str = "http://localhost:11434",
                 use_reranker: bool = False,
                 reranker_model: str = "BAAI/bge-reranker-v2-m3",
                 check_ollama_availability: bool = True,
                 ollama_options: Dict[str, Any] = None,
                 min_vram_mb: int = 500,
                 read_only: bool = False):
        """
        Инициализирует адаптер для Qdrant.

        Args:
            host: Хост Qdrant сервера
            port: Порт Qdrant сервера
            collection_name: Имя коллекции в Qdrant
            embeddings_type: Тип эмбеддингов
            model_name: Название модели эмбеддингов
            device: Устройство для вычислений
            ollama_url: URL для Ollama API
            use_reranker: Использовать ре-ранкер для уточнения результатов
            reranker_model: Название модели ре-ранкера
            check_ollama_availability: Проверять ли доступность Ollama при инициализации
            ollama_options: Опции для Ollama API
            min_vram_mb: Минимальное количество свободной VRAM в МБ для использования GPU
            read_only: Только для чтения (без обработки документов)
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embeddings_type = embeddings_type
        self.model_name = model_name
        self.device = device
        self.ollama_url = ollama_url
        self.min_vram_mb = min_vram_mb
        self.read_only = read_only

        # Получаем опции из OllamaEmbeddings
        from ppee_analyzer.vector_store.ollama_embeddings import OllamaEmbeddings
        options = OllamaEmbeddings.get_default_options()

        # Обновляем опции, если они переданы
        if ollama_options:
            options.update(ollama_options)

        # Инициализируем QdrantManager
        self.qdrant_manager = QdrantManager(
            collection_name=collection_name,
            host=host,
            port=port,
            embeddings_type=embeddings_type,
            model_name=model_name,
            device=device,
            ollama_url=ollama_url,
            check_availability=check_ollama_availability,
            ollama_options=options
        )

        # Инициализация ре-ранкера (при необходимости)
        self.use_reranker = use_reranker
        if use_reranker:
            try:
                # Инициализируем ре-ранкер с параметром min_vram_mb
                self.reranker = BGEReranker(
                    model_name=reranker_model,
                    device=device,
                    min_vram_mb=min_vram_mb
                )
                logger.info(f"Ре-ранкер инициализирован с моделью {reranker_model}")
            except Exception as e:
                logger.error(f"Ошибка при инициализации ре-ранкера: {str(e)}")
                self.use_reranker = False

        logger.info(f"QdrantAdapter инициализирован для коллекции {collection_name} на {host}:{port}")

        # Проверяем доступность семантического чанкера только если не read_only
        self.semantic_chunker_available = True
        if not read_only:
            try:
                from ppee_analyzer.semantic_chunker import SemanticChunker
                logger.info("Семантический чанкер доступен и будет использован для обработки документов")
            except ImportError:
                logger.warning("Модуль semantic_chunker не найден. Обработка документов будет недоступна.")
                self.semantic_chunker_available = False

    def _ensure_collection_exists(self):
        """Проверяет существование коллекции и создает при необходимости"""
        collections = self.qdrant_manager.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if self.collection_name not in collection_names:
            logger.info(f"Создание коллекции {self.collection_name}")
            self.qdrant_manager.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.qdrant_manager.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Коллекция {self.collection_name} успешно создана")

            # Создаем индексы для ускорения фильтрации
            self.qdrant_manager.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.application_id",
                field_schema="keyword"
            )
            self.qdrant_manager.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="metadata.content_type",
                field_schema="keyword"
            )

            # Создаем полнотекстовый индекс для поддержки текстового поиска
            self.qdrant_manager.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_content",
                field_schema="text"
            )

            logger.info(f"Индексы созданы успешно")
        else:
            logger.info(f"Коллекция {self.collection_name} уже существует")

            # Проверяем, существует ли полнотекстовый индекс
            try:
                collection_info = self.qdrant_manager.client.get_collection(collection_name=self.collection_name)

                # Проверяем, есть ли индекс для page_content
                text_index_exists = False
                if hasattr(collection_info, 'payload_schema') and collection_info.payload_schema:
                    text_index_exists = 'page_content' in collection_info.payload_schema

                # Если индекса нет, создаем его
                if not text_index_exists:
                    logger.info(f"Создание полнотекстового индекса для page_content")
                    self.qdrant_manager.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="page_content",
                        field_schema="text"
                    )
                    logger.info(f"Полнотекстовый индекс создан успешно")
            except Exception as e:
                logger.warning(f"Ошибка при проверке индексов: {str(e)}")

    def _process_document_with_semantic_chunker(self, document_path: str, application_id: str) -> List[Document]:
        """
        Обрабатывает документ с использованием семантического разделения.

        Args:
            document_path: Путь к документу
            application_id: ID заявки

        Returns:
            List[Document]: Список документов для индексации
        """
        # Проверяем доступность семантического чанкера перед обработкой
        if not self.semantic_chunker_available:
            raise ImportError("Модуль semantic_chunker не найден. Требуется для обработки документов.")

        try:
            # Импортируем SemanticChunker
            from ppee_analyzer.semantic_chunker import SemanticChunker

            # Определяем, использовать ли GPU
            use_gpu = None
            if hasattr(self, 'device') and self.device.lower() == 'cuda':
                use_gpu = True

            logger.info(f"Обработка документа с использованием семантического чанкера: {document_path}")

            # Получаем расширение файла
            _, ext = os.path.splitext(document_path)
            ext = ext.lower()

            # Проверяем, что это PDF
            if ext == '.pdf':
                # Инициализируем семантический чанкер
                chunker = SemanticChunker(use_gpu=use_gpu)

                # Шаг 1: Извлекаем смысловые блоки
                logger.info("Извлечение смысловых блоков...")
                chunks = chunker.extract_chunks(document_path)
                logger.info(f"Найдено {len(chunks)} начальных блоков")

                # Шаг 2: Обрабатываем таблицы
                logger.info("Обработка и объединение таблиц...")
                processed_chunks = chunker.post_process_tables(chunks)
                logger.info(f"После обработки таблиц: {len(processed_chunks)} блоков")

                # Шаг 3: Группируем короткие блоки
                logger.info("Группировка коротких блоков...")
                grouped_chunks = chunker.group_semantic_chunks(processed_chunks)
                logger.info(f"После группировки: {len(grouped_chunks)} финальных блоков")

                # Создаем идентификатор документа на основе имени файла
                document_id = f"doc_{os.path.basename(document_path).replace(' ', '_').replace('.', '_')}"
                document_name = os.path.basename(document_path)

                # Преобразуем в документы LangChain
                documents = []
                for i, chunk in enumerate(grouped_chunks):
                    # Создаем метаданные
                    metadata = {
                        "application_id": application_id,
                        "document_id": document_id,
                        "document_name": document_name,
                        "content_type": chunk.get("type", "unknown"),
                        "chunk_index": i,
                        "section": chunk.get("heading", "Не определено"),
                    }

                    # Добавляем информацию о странице
                    if chunk.get("page"):
                        metadata["page_number"] = chunk.get("page")

                    # Добавляем информацию о таблице
                    if chunk.get("type") == "table":
                        metadata["table_id"] = chunk.get("table_id")

                        # Если есть информация о нескольких страницах
                        if chunk.get("pages"):
                            metadata["pages"] = chunk.get("pages")

                    # Создаем документ
                    documents.append(Document(
                        page_content=chunk.get("content", ""),
                        metadata=metadata
                    ))

                return documents
            else:
                # Для других форматов временно возвращаем ошибку
                error_msg = f"Формат {ext} пока не поддерживается. Поддерживается только PDF."
                logger.error(error_msg)
                raise ValueError(error_msg)

        except Exception as e:
            logger.exception(f"Ошибка при обработке документа через семантический чанкер: {str(e)}")
            raise RuntimeError(f"Не удалось обработать документ: {str(e)}")

    def index_document(self,
                       application_id: str,
                       document_path: str,
                       delete_existing: bool = False) -> Dict[str, Any]:
        """
        Индексирует документ в Qdrant.

        Args:
            application_id: ID заявки
            document_path: Путь к документу
            delete_existing: Удалить существующие данные заявки

        Returns:
            Dict[str, Any]: Результаты индексации
        """
        try:
            # Проверяем, что файл существует
            if not os.path.exists(document_path):
                error_msg = f"Документ не найден по пути: {document_path}"
                logger.error(error_msg)
                return {
                    "application_id": application_id,
                    "document_path": document_path,
                    "error": error_msg,
                    "status": "error"
                }

            logger.info(f"Начало индексации документа {document_path} для заявки {application_id}")
            logger.info(f"Абсолютный путь к файлу: {os.path.abspath(document_path)}")
            logger.info(f"Размер файла: {os.path.getsize(document_path)} байт")

            # Проверяем расширение файла
            _, ext = os.path.splitext(document_path)
            ext = ext.lower()
            logger.info(f"Расширение файла: {ext}")

            # Обрабатываем документ через семантический чанкер
            chunks = self._process_document_with_semantic_chunker(document_path, application_id)
            logger.info(f"Документ успешно обработан: получено {len(chunks)} фрагментов")

            # Если нужно, удаляем существующие данные заявки
            if delete_existing:
                deleted_count = self.qdrant_manager.delete_application(application_id)
                logger.info(f"Удалено {deleted_count} существующих документов для заявки {application_id}")

            # Собираем статистику по типам фрагментов
            content_types = {}
            for chunk in chunks:
                content_type = chunk.metadata["content_type"]
                content_types[content_type] = content_types.get(content_type, 0) + 1

            # Индексируем фрагменты
            logger.info(f"Индексация {len(chunks)} фрагментов")
            indexed_count = self.qdrant_manager.add_documents(chunks)
            logger.info(f"Проиндексировано {indexed_count} фрагментов")

            # Формируем результат
            result = {
                "application_id": application_id,
                "document_path": document_path,
                "processing_path": document_path,
                "total_chunks": len(chunks),
                "indexed_count": indexed_count,
                "content_types": content_types,
                "status": "success"
            }

            return result

        except Exception as e:
            logger.exception(f"Ошибка при индексации документа: {str(e)}")
            return {
                "application_id": application_id,
                "document_path": document_path,
                "error": str(e),
                "status": "error"
            }

    def index_document_with_progress(self,
                                     application_id: str,
                                     document_path: str,
                                     delete_existing: bool = False,
                                     progress_callback=None) -> Dict[str, Any]:
        """
        Индексирует документ в Qdrant с отслеживанием прогресса.

        Args:
            application_id: ID заявки
            document_path: Путь к документу
            delete_existing: Удалить существующие данные заявки
            progress_callback: Функция обратного вызова для обновления прогресса

        Returns:
            Dict[str, Any]: Результаты индексации
        """
        try:
            # Проверяем, что файл существует
            if not os.path.exists(document_path):
                error_msg = f"Документ не найден по пути: {document_path}"
                logger.error(error_msg)
                return {
                    "application_id": application_id,
                    "document_path": document_path,
                    "error": error_msg,
                    "status": "error"
                }

            # Обновляем прогресс
            if progress_callback:
                progress_callback(20, 'convert', 'Начало обработки документа...')

            # Проверяем расширение файла
            _, ext = os.path.splitext(document_path)
            ext = ext.lower()

            # Для всех форматов используем семантический чанкер
            if progress_callback:
                progress_callback(25, 'convert', 'Начало семантического разделения документа...')

            # Обрабатываем документ через семантический чанкер
            chunks = self._process_document_with_semantic_chunker(document_path, application_id)
            logger.info(f"Документ успешно обработан: получено {len(chunks)} фрагментов")

            if progress_callback:
                progress_callback(40, 'split', 'Семантическое разделение завершено успешно')

            # Если нужно, удаляем существующие данные заявки
            if delete_existing:
                deleted_count = self.qdrant_manager.delete_application(application_id)
                logger.info(f"Удалено {deleted_count} существующих документов для заявки {application_id}")

            # Собираем статистику по типам фрагментов
            content_types = {}
            for chunk in chunks:
                content_type = chunk.metadata["content_type"]
                content_types[content_type] = content_types.get(content_type, 0) + 1

            # Обновляем прогресс
            if progress_callback:
                progress_callback(50, 'index', f'Начало индексации {len(chunks)} фрагментов...')

            # Индексируем фрагменты с отслеживанием прогресса
            total_chunks = len(chunks)
            batch_size = 20

            # Индексируем фрагменты пакетами
            for i in range(0, total_chunks, batch_size):
                end_idx = min(i + batch_size, total_chunks)
                batch = chunks[i:end_idx]

                # Добавляем пакет в индекс
                self.qdrant_manager.add_documents(batch)

                # Рассчитываем и обновляем прогресс
                progress = 50 + int(45 * (end_idx / total_chunks))
                logger.info(f"Индексация партии {i + 1}-{end_idx} из {total_chunks}")

                if progress_callback:
                    progress_callback(
                        progress,
                        'index',
                        f'Индексация фрагментов: {end_idx}/{total_chunks}...'
                    )

            logger.info(f"Проиндексировано {total_chunks} фрагментов")

            # Финальное обновление прогресса
            if progress_callback:
                progress_callback(95, 'complete', 'Завершение индексации...')

            # Формируем результат
            result = {
                "application_id": application_id,
                "document_path": document_path,
                "processing_path": document_path,
                "total_chunks": total_chunks,
                "indexed_count": total_chunks,
                "content_types": content_types,
                "status": "success"
            }

            return result

        except Exception as e:
            logger.exception(f"Ошибка при индексации документа: {str(e)}")

            # Обновляем прогресс с информацией об ошибке
            if progress_callback:
                progress_callback(0, 'error', f'Ошибка индексации: {str(e)}')

            return {
                "application_id": application_id,
                "document_path": document_path,
                "error": str(e),
                "status": "error"
            }

    def search(self,
               application_id: str,
               query: str,
               limit: int = 5,
               rerank_limit: int = None,
               use_reranker: bool = None) -> List[Dict[str, Any]]:
        """
        Выполняет семантический поиск с опциональным ре-ранкингом.

        Args:
            application_id: ID заявки
            query: Поисковый запрос
            limit: Количество результатов
            rerank_limit: Количество документов для ре-ранкинга (None - все найденные)
            use_reranker: Переопределение параметра self.use_reranker

        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        try:
            # Определяем, использовать ли ререйтинг (приоритет у переданного параметра)
            apply_reranker = use_reranker if use_reranker is not None else self.use_reranker

            logger.info(f"Выполнение поиска '{query}' для заявки {application_id} (ререйтинг: {apply_reranker})")

            # Увеличиваем limit для ре-ранкинга
            search_limit = limit
            if apply_reranker and rerank_limit is None:
                # Получаем больше результатов, чтобы ре-ранкер мог выбрать лучшие
                search_limit = max(limit * 3, 20)
            elif rerank_limit is not None:
                search_limit = rerank_limit

            # Выполняем поиск
            docs = self.qdrant_manager.search(
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
                    "score": doc.metadata.get('score', 0.0),  # Оценка векторного поиска
                    "search_type": "vector"
                })

            # Применяем ре-ранкинг, если он включен
            if apply_reranker and hasattr(self, 'reranker') and results:
                logger.info(f"Применение ре-ранкинга к {len(results)} результатам")
                try:
                    max_retries = 2  # Максимальное количество попыток ререйтинга
                    reranked_results = None

                    for attempt in range(max_retries):
                        try:
                            # Выполняем ре-ранкинг
                            reranked_results = self.reranker.rerank(
                                query=query,
                                documents=results,
                                top_k=limit,
                                text_key="text"
                            )
                            # Успешно получили результаты, выходим из цикла
                            break
                        except Exception as rerank_error:
                            # Если это последняя попытка, сохраняем ошибку для последующей обработки
                            if attempt == max_retries - 1:
                                logger.error(
                                    f"Ошибка при ререйтинге (попытка {attempt + 1}/{max_retries}): {str(rerank_error)}")
                                raise rerank_error
                            # Если это не последняя попытка, логируем и продолжаем
                            logger.warning(
                                f"Ошибка при ререйтинге (попытка {attempt + 1}/{max_retries}): {str(rerank_error)}. Повторяем...")

                    # Очищаем ресурсы после использования ререйтинга
                    self.cleanup()

                    # Если успешно получили результаты ререйтинга, возвращаем их
                    if reranked_results:
                        return reranked_results[:limit]

                except Exception as e:
                    logger.error(f"Ошибка при ререйтинге: {str(e)}, возвращаем исходные результаты")
                    # Освобождаем ресурсы даже при ошибке
                    self.cleanup()
                    return results[:limit]
            else:
                return results[:limit]
        except Exception as e:
            logger.error(f"Ошибка при поиске: {str(e)}")
            # В случае ошибки тоже освобождаем ресурсы
            if self.use_reranker:
                self.cleanup()
            return []

    def text_search(self,
                    application_id: str,
                    query: str,
                    limit: int = 5) -> List[Dict[str, Any]]:
        """
        Выполняет полнотекстовый поиск.

        Args:
            application_id: ID заявки
            query: Поисковый запрос
            limit: Количество результатов

        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        try:
            logger.info(f"Выполнение текстового поиска '{query}' для заявки {application_id}")

            # Создаем комбинированный фильтр для заявки и текстового поиска
            combined_filter = models.Filter(
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

            # Получаем результаты через scroll
            results = self.qdrant_manager.client.scroll(
                collection_name=self.qdrant_manager.collection_name,
                scroll_filter=combined_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )[0]

            # Преобразуем результаты
            formatted_results = []
            for i, point in enumerate(results):
                if "payload" in point.__dict__:
                    # Получаем текст и метаданные
                    text = point.payload.get("page_content", "")
                    metadata = point.payload.get("metadata", {})

                    # Добавляем искусственную оценку для текстового поиска
                    score = 1.0 - (i * 0.05)
                    if score < 0.1:
                        score = 0.1

                    formatted_results.append({
                        "text": text,
                        "metadata": metadata,
                        "score": score,
                        "search_type": "text"
                    })

            return formatted_results[:limit]

        except Exception as e:
            logger.error(f"Ошибка при текстовом поиске: {str(e)}")
            return []

    def hybrid_search(self,
                      application_id: str,
                      query: str,
                      limit: int = 5,
                      vector_weight: float = 0.5,
                      text_weight: float = 0.5,
                      use_reranker: bool = False) -> List[Dict[str, Any]]:
        """
        Выполняет гибридный поиск (комбинация векторного и текстового).

        Args:
            application_id: ID заявки
            query: Поисковый запрос
            limit: Количество результатов
            vector_weight: Вес векторного поиска (от 0 до 1)
            text_weight: Вес текстового поиска (от 0 до 1)
            use_reranker: Применять ререйтинг к результатам поиска

        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        try:
            logger.info(f"Выполнение гибридного поиска '{query}' для заявки {application_id}")
            logger.info(
                f"Параметры: vector_weight={vector_weight}, text_weight={text_weight}, use_reranker={use_reranker}")

            # Получаем результаты векторного поиска (без ререйтинга)
            vector_results = self.search(
                application_id=application_id,
                query=query,
                limit=limit * 2,  # Запрашиваем больше результатов для объединения
                rerank_limit=None,
                use_reranker=False  # Важно: отключаем ререйтинг для векторного поиска
            )

            # Получаем результаты текстового поиска
            text_results = self.text_search(
                application_id=application_id,
                query=query,
                limit=limit * 2
            )

            # Объединяем результаты
            combined_results = self._combine_results(vector_results, text_results,
                                                     vector_weight, text_weight, limit)

            # Проверка результатов перед ререйтингом
            for i, doc in enumerate(combined_results[:3]):  # Логируем первые 3 для примера
                logger.info(f"Документ {i}: наличие поля 'text': {'text' in doc}, "
                            f"длина текста: {len(doc.get('text', ''))}")

            # Применяем ререйтинг к объединенным результатам, если нужно
            if use_reranker and self.use_reranker and combined_results:
                logger.info(f"Применение ререйтинга к результатам гибридного поиска")
                try:
                    reranked_results = self.reranker.rerank(
                        query=query,
                        documents=combined_results,
                        top_k=limit,
                        text_key="text"
                    )
                    # Очищаем ресурсы после использования ререйтинга
                    self.cleanup()
                    return reranked_results[:limit]
                except Exception as e:
                    logger.error(
                        f"Ошибка при ререйтинге гибридных результатов: {str(e)}, возвращаем исходные результаты")
                    return combined_results[:limit]

            return combined_results

        except Exception as e:
            logger.error(f"Ошибка при гибридном поиске: {str(e)}")
            # В случае ошибки тоже освобождаем ресурсы
            if use_reranker and self.use_reranker:
                self.cleanup()
            return []

    def _combine_results(self,
                         vector_results: List[Dict],
                         text_results: List[Dict],
                         vector_weight: float,
                         text_weight: float,
                         limit: int) -> List[Dict[str, Any]]:
        """
        Объединяет результаты векторного и текстового поиска.

        Args:
            vector_results: Результаты векторного поиска
            text_results: Результаты текстового поиска
            vector_weight: Вес векторного поиска
            text_weight: Вес текстового поиска
            limit: Максимальное количество результатов

        Returns:
            List[Dict[str, Any]]: Объединенные результаты
        """
        # Нормализуем веса
        total_weight = vector_weight + text_weight
        vector_weight = vector_weight / total_weight
        text_weight = text_weight / total_weight

        # Создаем словарь для объединения результатов
        results_dict = {}

        # Добавляем векторные результаты
        for doc in vector_results:
            doc_key = self._get_document_key(doc)
            score = doc.get("score", 0.0) * vector_weight

            results_dict[doc_key] = {
                "doc": doc,
                "score": score,
                "search_type": "hybrid"
            }

        # Добавляем текстовые результаты
        for doc in text_results:
            doc_key = self._get_document_key(doc)
            text_score = doc.get("score", 0.0) * text_weight

            if doc_key in results_dict:
                # Если документ уже есть, обновляем оценку
                results_dict[doc_key]["score"] += text_score
            else:
                # Иначе добавляем новый документ
                results_dict[doc_key] = {
                    "doc": doc,
                    "score": text_score,
                    "search_type": "hybrid"
                }

        # Сортируем и ограничиваем количество
        sorted_results = sorted(results_dict.values(),
                                key=lambda x: x["score"], reverse=True)[:limit]

        # Логируем количество результатов
        logger.info(f"Объединенные результаты: найдено {len(sorted_results)} элементов после объединения и сортировки")

        # Конвертируем обратно в список результатов
        combined_results = []
        for item in sorted_results:
            doc = item["doc"].copy()
            doc["score"] = item["score"]
            doc["search_type"] = "hybrid"

            # ВАЖНО: Убедимся, что есть поле text для ререйтинга
            if "text" not in doc and "content" in doc:
                doc["text"] = doc["content"]
            elif "text" not in doc and "page_content" in doc:
                doc["text"] = doc["page_content"]

            combined_results.append(doc)

        return combined_results

    def _get_document_key(self, doc: Dict) -> str:
        """
        Создает уникальный ключ для документа.

        Args:
            doc: Документ

        Returns:
            str: Уникальный ключ
        """
        metadata = doc.get("metadata", {})

        # Создаем составной ключ из доступных метаданных
        key_parts = []

        if "document_id" in metadata:
            key_parts.append(f"doc:{metadata['document_id']}")

        if "chunk_index" in metadata:
            key_parts.append(f"chunk:{metadata['chunk_index']}")

        if "page_number" in metadata:
            key_parts.append(f"page:{metadata['page_number']}")

        if key_parts:
            return "|".join(key_parts)

        # Запасной вариант - хеш текста
        return str(hash(doc.get("text", "")))

    def smart_search(self,
                     application_id: str,
                     query: str,
                     limit: int = 5,
                     use_reranker: bool = False,
                     rerank_limit: int = None,
                     vector_weight: float = 0.5,
                     text_weight: float = 0.5,
                     hybrid_threshold: int = 10) -> List[Dict[str, Any]]:
        """
        Умный поиск, выбирающий метод в зависимости от длины запроса:
        - Для коротких запросов (< hybrid_threshold) - гибридный поиск
        - Для длинных запросов - векторный поиск

        Args:
            application_id: ID заявки
            query: Поисковый запрос
            limit: Количество результатов
            use_reranker: Использовать ли ререйтинг
            rerank_limit: Количество документов для ререйтинга
            vector_weight: Вес векторного поиска для гибридного поиска
            text_weight: Вес текстового поиска для гибридного поиска
            hybrid_threshold: Порог длины запроса для гибридного поиска

        Returns:
            List[Dict[str, Any]]: Результаты поиска
        """
        # Выбираем метод поиска в зависимости от длины запроса
        if len(query) < hybrid_threshold:
            logger.info(f"Запрос '{query}' короткий (<{hybrid_threshold} символов), "
                        f"используем гибридный поиск с ререйтингом={use_reranker}")
            results = self.hybrid_search(
                application_id=application_id,
                query=query,
                limit=limit,
                vector_weight=vector_weight,
                text_weight=text_weight,
                use_reranker=use_reranker
            )
        else:
            logger.info(f"Запрос '{query}' длинный (>={hybrid_threshold} символов), "
                        f"используем векторный поиск с ререйтингом={use_reranker}")
            # ИСПРАВЛЕНИЕ: Явно передаем use_reranker!
            results = self.search(
                application_id=application_id,
                query=query,
                limit=limit,
                rerank_limit=rerank_limit,
                use_reranker=use_reranker  # ВАЖНО: Явная передача параметра use_reranker
            )

        return results

    def delete_application_data(self, application_id: str) -> bool:
        """
        Удаляет данные заявки из хранилища.

        Args:
            application_id: ID заявки

        Returns:
            bool: Успешность операции
        """
        try:
            deleted_count = self.qdrant_manager.delete_application(application_id)
            logger.info(f"Удалено {deleted_count} документов для заявки {application_id}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении данных заявки: {str(e)}")
            return False

    def cleanup(self):
        """
        Освобождает ресурсы после использования.
        Особенно важно для освобождения памяти GPU после поиска с ререйтингом.
        """
        if self.use_reranker and hasattr(self, 'reranker'):
            logger.info("Освобождение ресурсов ререйтера...")
            try:
                # Вызываем метод cleanup у ререйтера
                if hasattr(self.reranker, 'cleanup'):
                    self.reranker.cleanup()
            except Exception as e:
                logger.error(f"Ошибка при освобождении ресурсов ререйтера: {str(e)}")