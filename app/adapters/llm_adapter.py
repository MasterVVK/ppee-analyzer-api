import logging
import requests
import json
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Интерфейс для провайдеров LLM"""

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Возвращает список доступных моделей"""
        pass

    @abstractmethod
    def process_query(self,
                      model_name: str,
                      prompt: str,
                      context: str,
                      parameters: Dict[str, Any],
                      query: str = None) -> str:
        """Обрабатывает запрос к LLM"""
        pass


class OllamaLLMProvider(LLMProvider):
    """Реализация провайдера LLM через Ollama API"""

    # Максимальный допустимый размер контекста для Ollama
    MAX_OLLAMA_CONTEXT = 32768

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Инициализирует провайдер Ollama.

        Args:
            base_url: URL для Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self._models_cache = {}  # Кэш для информации о моделях
        logger.info(f"OllamaLLMProvider инициализирован с URL: {self.base_url}")

    def get_available_models(self) -> List[str]:
        """
        Возвращает список доступных моделей в Ollama.

        Returns:
            List[str]: Список имен моделей
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = []
            for model in response.json().get("models", []):
                models.append(model.get("name"))

            logger.info(f"Получено {len(models)} доступных моделей из Ollama")
            return models
        except Exception as e:
            logger.error(f"Ошибка при получении списка моделей из Ollama: {str(e)}")
            return []

    def get_model_info(self, model_name: str, refresh: bool = False) -> Dict[str, Any]:
        """
        Получает информацию о модели из Ollama API.

        Args:
            model_name: Название модели
            refresh: Принудительно обновить кэш

        Returns:
            Dict[str, Any]: Информация о модели
        """
        # Проверяем кэш, если обновление не требуется
        if not refresh and model_name in self._models_cache:
            return self._models_cache[model_name]

        try:
            # Вариант 1: Используем POST-запрос с JSON-телом
            # Этот вариант более надежен для имен с спецсимволами
            url = f"{self.base_url}/api/show"

            response = requests.post(
                url,
                json={"name": model_name, "verbose": True},  # Добавляем verbose=True для получения полной информации
                timeout=5
            )

            # Если POST не работает, попробуем другие методы
            if response.status_code != 200:
                logger.info(f"POST запрос не сработал, пробуем другие варианты для {model_name}")

                # Вариант 2: Получаем список всех моделей и ищем нужную
                models_response = requests.get(f"{self.base_url}/api/tags")
                models_response.raise_for_status()

                models_data = models_response.json().get("models", [])
                found_model = None

                # Ищем модель по имени (без учета регистра)
                for model in models_data:
                    if model.get("name", "").lower() == model_name.lower():
                        found_model = model
                        break

                if found_model:
                    logger.info(f"Модель {model_name} найдена в списке моделей")
                    self._models_cache[model_name] = found_model
                    return found_model
                else:
                    # Если модель не найдена, создаем базовую информацию
                    logger.warning(f"Модель {model_name} не найдена в API Ollama")
                    default_info = {
                        "name": model_name,
                        "not_found": True,
                        # Устанавливаем значения по умолчанию
                        "parameters": {
                            "context_length": self._get_default_context_length(model_name)
                        }
                    }
                    self._models_cache[model_name] = default_info
                    return default_info

            # Если запрос успешный, обрабатываем результат
            model_info = response.json()
            logger.info(f"Получена информация о модели {model_name}")
            logger.debug(f"Структура ответа: {json.dumps(model_info, indent=2, default=str)}")

            # Сохраняем в кэш
            self._models_cache[model_name] = model_info
            return model_info
        except Exception as e:
            logger.error(f"Ошибка при получении информации о модели {model_name}: {str(e)}")
            # Возвращаем информацию с отметкой об ошибке и значениями по умолчанию
            default_info = {
                "name": model_name,
                "error": str(e),
                "parameters": {
                    "context_length": self._get_default_context_length(model_name)
                }
            }
            self._models_cache[model_name] = default_info
            return default_info

    def _get_default_context_length(self, model_name: str) -> int:
        """
        Возвращает размер контекста по умолчанию для модели на основе ее названия.

        Args:
            model_name: Название модели

        Returns:
            int: Размер контекста по умолчанию
        """
        model_name_lower = model_name.lower()

        if "gemma3:27b" in model_name_lower:
            return 8192
        elif "gemma3:7b" in model_name_lower:
            return 8192
        elif "gemma3:2b" in model_name_lower:
            return 8192
        elif "llama3:70b" in model_name_lower:
            return 8192
        elif "llama3:8b" in model_name_lower:
            return 8192
        elif "mistral" in model_name_lower:
            return 8192
        elif "mixtral" in model_name_lower:
            return 32768
        elif "phi3" in model_name_lower:
            return 4096

        # Значение по умолчанию
        return 4096

    def _parse_parameters_string(self, params_str: str) -> Dict[str, Any]:
        """
        Парсит строку параметров в словарь.

        Args:
            params_str: Строка параметров в формате "key value\nkey value"

        Returns:
            Dict[str, Any]: Словарь параметров
        """
        params = {}
        if not params_str:
            return params

        # Разбиваем на строки
        lines = params_str.strip().split('\n')
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                # Пытаемся конвертировать в число, если возможно
                try:
                    if '.' in value:
                        params[key] = float(value)
                    else:
                        params[key] = int(value)
                except ValueError:
                    # Если не число, сохраняем как строку, удаляя кавычки
                    params[key] = value.strip('"').strip("'")
            elif len(parts) == 1:
                # Параметр без значения (флаг)
                params[parts[0]] = True

        return params

    def get_context_length(self, model_name: str) -> int:
        """
        Получает максимальный размер контекста для модели.

        Args:
            model_name: Название модели

        Returns:
            int: Максимальный размер контекста
        """
        # Получаем информацию о модели
        model_info = self.get_model_info(model_name)

        logger.debug(f"Определение context_length для модели {model_name}")

        # Проверяем наличие ошибок или отсутствие модели
        if model_info.get("not_found") or model_info.get("error"):
            context_length = None
            if isinstance(model_info.get("parameters"), dict):
                context_length = model_info.get("parameters", {}).get("context_length")

            if context_length is not None:
                try:
                    return int(context_length)
                except (ValueError, TypeError):
                    pass

            default_length = self._get_default_context_length(model_name)
            logger.info(f"Используем размер контекста по умолчанию: {default_length}")
            return default_length

        # Вариант 1: Новый путь для Ollama >= 0.1.18 - через model_info
        if 'model_info' in model_info and isinstance(model_info['model_info'], dict):
            model_specific_info = model_info['model_info']

            # Для моделей LLaMA
            if 'llama.context_length' in model_specific_info:
                try:
                    context_length = int(model_specific_info['llama.context_length'])
                    logger.info(f"Найден context_length в model_info.llama.context_length: {context_length}")
                    return context_length
                except (ValueError, TypeError) as e:
                    logger.warning(f"Ошибка при конвертации llama.context_length: {e}")

            # Для других моделей могут быть другие пути
            for key in model_specific_info:
                if key.endswith('.context_length'):
                    try:
                        context_length = int(model_specific_info[key])
                        logger.info(f"Найден context_length в model_info.{key}: {context_length}")
                        return context_length
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Ошибка при конвертации {key}: {e}")

        # Вариант 2: Прямой параметр context_length
        if 'context_length' in model_info:
            try:
                context_length = int(model_info['context_length'])
                logger.info(f"Найден прямой параметр context_length: {context_length}")
                return context_length
            except (ValueError, TypeError) as e:
                logger.warning(f"Ошибка при конвертации context_length: {e}")

        # Вариант 3: Через parameters (в разных форматах)
        if 'parameters' in model_info:
            params = model_info['parameters']

            # Если parameters - словарь
            if isinstance(params, dict):
                for key in ['context_length', 'num_ctx', 'ctx_len', 'context_window']:
                    if key in params:
                        try:
                            context_length = int(params[key])
                            logger.info(f"Найден {key} в parameters (dict): {context_length}")
                            return context_length
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Ошибка при конвертации {key} из parameters (dict): {e}")

            # Если parameters - строка
            elif isinstance(params, str):
                # Парсим строку в словарь
                parsed_params = self._parse_parameters_string(params)

                # Проверяем разные возможные имена параметра
                for key in ['context_length', 'num_ctx', 'ctx_len', 'context_window']:
                    if key in parsed_params:
                        try:
                            context_length = int(parsed_params[key])
                            logger.info(f"Найден {key} в parameters (string): {context_length}")
                            return context_length
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Ошибка при конвертации {key} из parameters (string): {e}")

                # Если парсинг не помог, ищем напрямую в строке
                for key in ['context_length', 'num_ctx', 'ctx_len', 'context_window']:
                    match = re.search(fr'{key}\s+(\d+)', params, re.IGNORECASE)
                    if match:
                        try:
                            context_length = int(match.group(1))
                            logger.info(f"Найден {key} через regex в parameters (string): {context_length}")
                            return context_length
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Ошибка при конвертации {key} через regex: {e}")

        # Вариант 4: Через modelfile
        if 'modelfile' in model_info and isinstance(model_info['modelfile'], str):
            modelfile = model_info['modelfile']

            # Ищем все возможные параметры контекста в modelfile
            for key in ['context_length', 'num_ctx', 'ctx_len', 'context_window']:
                match = re.search(fr'PARAMETER\s+{key}\s+(\d+)', modelfile, re.IGNORECASE)
                if match:
                    try:
                        context_length = int(match.group(1))
                        logger.info(f"Найден {key} в modelfile: {context_length}")
                        return context_length
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Ошибка при конвертации {key} из modelfile: {e}")

        # Вариант 5: Через details, если есть
        if 'details' in model_info and isinstance(model_info['details'], dict):
            details = model_info['details']

            # Некоторые модели могут содержать информацию о контексте в details
            if 'context_length' in details:
                try:
                    context_length = int(details['context_length'])
                    logger.info(f"Найден context_length в details: {context_length}")
                    return context_length
                except (ValueError, TypeError) as e:
                    logger.warning(f"Ошибка при конвертации context_length из details: {e}")

        # Если ничего не нашли, используем значение по умолчанию
        default_length = self._get_default_context_length(model_name)
        logger.info(f"Не удалось найти context_length, используем значение по умолчанию: {default_length}")

        # Ограничиваем максимальный размер контекста
        if default_length > self.MAX_OLLAMA_CONTEXT:
            logger.warning(
                f"Значение context_length ({default_length}) превышает максимально допустимое "
                f"({self.MAX_OLLAMA_CONTEXT}), ограничиваем"
            )
            return self.MAX_OLLAMA_CONTEXT

        return default_length

    def process_query(self,
                      model_name: str,
                      prompt: str,
                      context: str,
                      parameters: Dict[str, Any],
                      query: str = None) -> str:
        """
        Обрабатывает запрос к LLM через Ollama API.

        Args:
            model_name: Название модели
            prompt: Шаблон промпта
            context: Контекст (результаты поиска)
            parameters: Параметры генерации
            query: Поисковый запрос (если нужно заменить {query} в шаблоне)

        Returns:
            str: Ответ модели
        """
        try:
            # Создаем полный промпт, заменяя плейсхолдеры
            full_prompt = prompt.replace("{context}", context)

            # Обязательно заменяем {query} на реальный запрос
            if query:
                full_prompt = full_prompt.replace("{query}", query)
            else:
                # Если query не передан, используем значение из параметра search_query
                search_query = parameters.get("search_query", "запрос")
                full_prompt = full_prompt.replace("{query}", search_query)

            # Получаем максимальный размер контекста для модели
            context_length = self.get_context_length(model_name)
            if context_length > 16384: context_length = 16384

            # Формируем параметры для запроса
            options = {
                # Явно указываем размер контекста, чтобы предотвратить обрезание
                "num_ctx": context_length,
                # Устанавливаем, сколько токенов нужно сохранить с начала промпта
                "num_keep": context_length,
                # Добавляем параметр для отключения режима think
                "raw_prompt": True,  # Использовать промпт как есть, без обработки
                "think": False  # Отключаем режим размышлений
            }

            # Добавляем температуру, если указана
            if "temperature" in parameters:
                options["temperature"] = float(parameters["temperature"])

            # Добавляем max_tokens, если указано
            if "max_tokens" in parameters:
                options["num_predict"] = int(parameters["max_tokens"])

            # Оценка длины промпта (примерно 1 токен = 4 символа)
            prompt_length = len(full_prompt) // 4
            logger.info(f"Отправка запроса к модели {model_name} через Ollama API")
            logger.info(f"Примерная длина промпта: ~{prompt_length} токенов")
            logger.info(
                f"Указанный в запросе размер контекста: {options['num_ctx']}, сохраняемый контекст: {options['num_keep']}")

            # Формируем запрос к Ollama API
            payload = {
                "model": model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": options,
                "keep_alive": "10s"
            }

            # Отправляем запрос
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=1200  # Увеличенный таймаут для сложных запросов
            )

            # Проверяем ответ
            if response.status_code != 200:
                error_msg = f"HTTP ошибка {response.status_code}: {response.text}"
                logger.error(error_msg)
                try:
                    error_json = response.json()
                    if "error" in error_json:
                        error_msg += f" - {error_json['error']}"
                except:
                    pass
                return f"Ошибка сети: {error_msg}"

            # Получаем ответ
            result = response.json()

            # Проверяем на наличие ошибок в ответе
            if "error" in result:
                error_msg = f"Ошибка Ollama API: {result.get('error')}"
                logger.error(error_msg)
                return f"Ошибка: {error_msg}"

            answer = result.get("response", "")

            # Проверяем, содержит ли ответ блоки <think>
            import re
            think_blocks = re.findall(r'<think>.*?</think>', answer, flags=re.DOTALL)

            # Если блоки найдены, удаляем их
            if think_blocks:
                logger.info(f"Обнаружено {len(think_blocks)} блоков <think> в ответе модели")
                # Удаляем весь текст между тегами <think> и </think>, включая сами теги
                cleaned_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
                # Убираем возможные двойные переносы строк, образовавшиеся после удаления блоков
                cleaned_answer = re.sub(r'\n\s*\n', '\n\n', cleaned_answer)
                # Обрезаем пробелы в начале и конце
                cleaned_answer = cleaned_answer.strip()

                # Если после очистки ответ пустой, возвращаем строку о проблеме
                if not cleaned_answer:
                    logger.warning("После удаления блоков <think> ответ стал пустым")
                    return "Информация не найдена"

                answer = cleaned_answer
            else:
                logger.info("Блоки <think> не обнаружены в ответе модели")

            # Выводим информацию о загрузке и генерации
            total_duration = result.get('total_duration', 0)
            load_duration = result.get('load_duration', 0)
            prompt_eval_count = result.get('prompt_eval_count', 0)
            eval_count = result.get('eval_count', 0)

            # Выводим полный ответ модели для отладки
            logger.info(f"Итоговый ответ от модели {model_name} длиной {len(answer)} символов")
            logger.info(f"Статистика генерации: prompt_tokens={prompt_eval_count}, output_tokens={eval_count}")

            if total_duration > 0:
                total_seconds = total_duration / 1e9  # Преобразуем наносекунды в секунды
                logger.info(f"Время выполнения: {total_seconds:.2f} сек (загрузка: {load_duration / 1e9:.2f} сек)")

            # Возвращаем обработанный ответ
            return answer

        except requests.exceptions.RequestException as e:
            error_msg = f"Ошибка HTTP при запросе к Ollama API: {str(e)}"
            logger.error(error_msg)
            return f"Ошибка сети: {error_msg}"
        except Exception as e:
            error_msg = f"Ошибка при обработке запроса через Ollama API: {str(e)}"
            logger.error(error_msg)
            return f"Ошибка: {error_msg}"