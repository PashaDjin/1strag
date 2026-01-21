"""
Streamlit UI + RAG-чат с retriever + llm напрямую.

Stage 3: Core RAG logic (функции без UI).
Stage 4: main() + Streamlit UI.
"""

import json
import os

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM


# --- Константы по умолчанию ---
DEFAULT_INDEX_DIR = "rag_index/"
DEFAULT_TOP_K = 20  # Увеличено: чанки 500 токенов → нужно больше для полного контекста
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"  # Лучший для русского. Альтернатива: qwen2.5:7b
ENABLE_QUERY_EXPANSION = True  # Расширение запроса синонимами

# Системный промпт с Chain-of-Thought
SYSTEM_PROMPT = """Ты — эксперт-аналитик. Твоя ЕДИНСТВЕННАЯ задача — извлекать информацию из контекста ниже.

⚠️ КРИТИЧЕСКИ ВАЖНО:
- Используй ТОЛЬКО информацию из контекста ниже
- НИКОГДА не используй свои знания — только цитируй книгу
- Если формула есть в контексте — цитируй её ДОСЛОВНО
- Если чего-то нет в контексте — НЕ придумывай, напиши "в книге не указано"

ШАГ 1 — ИЗВЛЕЧЕНИЕ: Перечитай КАЖДЫЙ фрагмент и выпиши ВСЕ термины/понятия по теме вопроса.
Важно: просмотри ВСЕ фрагменты, информация распределена между ними!

ШАГ 2 — АНАЛИЗ: Для каждого термина кратко объясни суть (по контексту).

ШАГ 3 — СИНТЕЗ: Объедини в структурированный ответ со списком.

ЗАПРЕЩЕНО:
❌ Придумывать формулы или определения
❌ Использовать информацию НЕ из контекста
❌ Останавливаться на 2-3 пунктах — ищи ВСЁ

Контекст из книг:
{context}

Вопрос: {question}

Ответ (СТРОГО по контексту выше):"""


# --- Вспомогательные функции для env ---

def get_env_int(name: str, default: int) -> int:
    """Безопасно читает int из env."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_str(name: str, default: str) -> str:
    """Читает строку из env с дефолтом."""
    return os.getenv(name, default)


# --- Загрузка конфига и индекса ---

def load_index_config(index_dir: str) -> dict | None:
    """
    Загружает config.json с параметрами сборки.
    Возвращает None если файла нет.
    """
    config_path = os.path.join(index_dir, "config.json")
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Проверяем только embed_model — остальное опционально
    if "embed_model" not in config:
        raise RuntimeError(
            "❌ В config.json отсутствует ключ 'embed_model'.\n"
            "   Пересоберите индекс: python rag_setup.py"
        )
    return config


def is_e5_model(model_name: str) -> bool:
    """Проверяет, является ли модель E5 (требует префиксы query:/passage:)."""
    return "e5" in model_name.lower()


class E5QueryEmbeddings(HuggingFaceEmbeddings):
    """
    Обёртка над HuggingFaceEmbeddings, добавляющая 'query: ' префикс.
    E5 модели обучались с префиксами — это критично для качества!
    """
    def embed_query(self, text: str) -> list[float]:
        """Добавляет query: префикс перед получением эмбеддинга запроса."""
        return super().embed_query(f"query: {text}")


def check_embed_model_mismatch(config: dict) -> bool:
    """
    Сравнивает config["embed_model"] с os.getenv("EMBED_MODEL").
    Если отличаются — возвращает True (показать warning).
    """
    env_model = os.getenv("EMBED_MODEL")
    if env_model is None:
        return False
    return env_model != config["embed_model"]


@st.cache_resource
def load_index(index_dir: str, embed_model: str):
    """
    Загружает FAISS индекс с диска.
    Embeddings создаются ЗДЕСЬ ОДИН РАЗ.
    Возвращает None если индекса нет.
    
    ВНИМАНИЕ: allow_dangerous_deserialization=True безопасно
    ТОЛЬКО для вашего собственного индекса. Не загружайте чужие индексы!
    """
    index_path = os.path.join(index_dir, "index.faiss")
    if not os.path.exists(index_path):
        return None
    
    # Для E5 моделей используем обёртку с query: префиксом
    if is_e5_model(embed_model):
        embeddings = E5QueryEmbeddings(model_name=embed_model)
    else:
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    
    vectorstore = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True  # Безопасно только для своего индекса!
    )
    return vectorstore


def get_retriever(vectorstore, top_k: int):
    """
    Создаёт retriever с заданным k.
    ВАЖНО: k задаётся при создании retriever, НЕ при вызове.
    """
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


# --- LLM ---

@st.cache_resource
def get_llm(model: str) -> OllamaLLM:
    """Создаёт LLM из Ollama."""
    # OLLAMA_BASE_URL позволяет использовать удалённую Ollama (например, через ngrok)
    base_url = get_env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    return OllamaLLM(
        base_url=base_url,
        model=model,
        temperature=0,
    )


def check_ollama_connection(llm: OllamaLLM) -> bool:
    """
    Проверяет доступность Ollama перед использованием.
    Используем простой GET запрос вместо invoke для скорости.
    """
    import urllib.request
    import urllib.error
    
    base_url = get_env_str("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


# --- История и промпт ---

def build_history_text(messages: list[dict], max_pairs: int = 3) -> str:
    """
    Конвертирует UI-историю в текст для промпта.
    Берёт только последние max_pairs пар (user, assistant).
    Если история пуста — возвращает пустую строку.
    """
    if not messages:
        return ""
    
    # Собираем пары user/assistant
    pairs = []
    i = 0
    while i < len(messages) - 1:
        if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
            pairs.append((messages[i]["content"], messages[i + 1]["content"]))
            i += 2
        else:
            i += 1
    
    if not pairs:
        return ""
    
    # Берём последние max_pairs
    pairs = pairs[-max_pairs:]
    
    # Форматируем
    lines = []
    for user_msg, assistant_msg in pairs:
        lines.append(f"Вопрос: {user_msg}")
        lines.append(f"Ответ: {assistant_msg}")
        lines.append("")  # Пустая строка между парами
    
    return "\n".join(lines).strip()


def build_full_question(user_question: str, messages: list[dict], max_pairs: int = 3) -> str:
    """
    Формирует полный вопрос с историей для retriever.
    """
    history_text = build_history_text(messages, max_pairs)
    if history_text:
        return f"Предыдущий диалог:\n{history_text}\n\nТекущий вопрос: {user_question}"
    return user_question


def format_context(docs: list) -> str:
    """
    Собирает контекст из документов с нумерацией "X из Y".
    Порядок: менее релевантные сначала, самый релевантный — в конце.
    (LLM лучше запоминают конец контекста — "recency bias")
    
    Поддерживает два формата metadata:
    - Новый (HybridChunker): section/headings
    - Старый (legacy): page_label/page
    """
    if not docs:
        return ""
    
    total = len(docs)
    # Переворачиваем: самый релевантный будет последним
    reversed_docs = list(reversed(docs))
    
    fragments = []
    for i, doc in enumerate(reversed_docs, 1):
        # Новый формат: section от HybridChunker
        section = doc.metadata.get("section")
        if section:
            location = section
        else:
            # Старый формат: page number
            page_label = doc.metadata.get("page_label")
            if not page_label:
                page_num = doc.metadata.get("page")
                page_label = str(page_num + 1) if page_num is not None else "?"
            location = f"стр. {page_label}"
        
        # "X из Y" создаёт ощущение чек-листа для LLM
        header = f"[Фрагмент {i} из {total}, {location}]"
        fragments.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(fragments)


def build_prompt(context: str, question: str) -> str:
    """Собирает финальный промпт для LLM."""
    return SYSTEM_PROMPT.format(context=context, question=question)


# --- Query Expansion ---

QUERY_EXPANSION_PROMPT = """Расширь поисковый запрос синонимами и связанными терминами.

Запрос: {question}

Добавь:
- Синонимы на русском
- Английские эквиваленты терминов
- Связанные понятия

Отвечай ТОЛЬКО списком слов через пробел, без пояснений.
Пример: "виды прибыли валовая прибыль gross profit чистая прибыль net income EBITDA retained earnings"

Расширенный запрос:"""


def expand_query(question: str, llm) -> str:
    """
    Расширяет запрос синонимами и связанными терминами.
    Помогает найти чанки с английскими терминами и синонимами.
    """
    prompt = QUERY_EXPANSION_PROMPT.format(question=question)
    try:
        expanded = llm.invoke(prompt)
        # Объединяем оригинальный вопрос с расширением
        result = f"{question} {expanded.strip()}"
        return result
    except Exception:
        return question  # Fallback к оригинальному вопросу


# --- Форматирование источников ---

def format_source(doc) -> str:
    """
    Форматирует источник.
    
    Новый формат с HybridChunker: "book.pdf [Глава 2 > Виды прибыли]"
    Старый формат с page: "book.pdf [стр. 23]"
    """
    filename = os.path.basename(doc.metadata.get("source", "unknown"))
    
    # Новый формат: headings от HybridChunker
    section = doc.metadata.get("section")
    if section:
        return f"{filename} [{section}]"
    
    # Старый формат: page number
    page_label = doc.metadata.get("page_label")
    if page_label:
        return f"{filename} [стр. {page_label}]"
    
    page_num = doc.metadata.get("page")
    if page_num is not None:
        return f"{filename} [стр. {page_num + 1}]"
    
    return filename


def format_sources(docs: list) -> list[str]:
    """
    Форматирует и дедуплицирует список источников.
    Сохраняет порядок первого появления.
    """
    # DEBUG: показать сколько чанков найдено
    print(f"[DEBUG] Найдено чанков: {len(docs)}")
    for i, doc in enumerate(docs):
        section = doc.metadata.get("section", "")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  [{i+1}] {section}: {preview}...")
    
    seen = set()
    result = []
    for doc in docs:
        source = format_source(doc)
        if source not in seen:
            seen.add(source)
            result.append(source)
    return result



# --- Главная функция ответа ---

def ask_question(
    retriever,
    llm,
    question: str,
    messages: list[dict],
) -> tuple[str, list]:
    """
    Отвечает на вопрос используя retriever и llm НАПРЯМУЮ.
    
    НЕ используем RetrievalQA chain!
    
    Возвращает (answer, docs) где docs — список Document для sources.
    """
    # 1. Формируем полный вопрос с историей
    full_question = build_full_question(question, messages)
    
    # 1.5 Query Expansion: расширяем запрос синонимами
    if ENABLE_QUERY_EXPANSION:
        search_query = expand_query(full_question, llm)
    else:
        search_query = full_question
    
    # 2. Получаем документы (широкий охват)
    # Основной метод — invoke, fallback — get_relevant_documents
    try:
        docs = retriever.invoke(search_query)
    except AttributeError:
        docs = retriever.get_relevant_documents(search_query)
    
    # Проверяем что получили список
    if not isinstance(docs, list):
        docs = list(docs) if docs else []
    
    # 3. Если нет документов — НЕ вызываем LLM
    if not docs:
        return "В книгах нет информации по этому вопросу.", []
    
    # 4. Формируем контекст и получаем ответ
    context = format_context(docs)
    prompt = build_prompt(context, full_question)
    answer = llm.invoke(prompt)
    
    # 5. Возвращаем ответ и документы
    return answer, docs


# --- Stage 4: Streamlit UI ---

def main():
    """Главная функция Streamlit приложения."""
    
    # Конфигурация страницы
    st.set_page_config(
        page_title="RAG чатбот по PDF",
        page_icon="📚",
        layout="wide",
    )
    
    # Заголовок
    st.title("📚 RAG чатбот по PDF")
    
    # Инициализация session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Получаем пути из env
    index_dir = get_env_str("INDEX_DIR", DEFAULT_INDEX_DIR)
    books_dir = get_env_str("BOOKS_DIR", "books/")
    top_k = get_env_int("TOP_K", DEFAULT_TOP_K)
    ollama_model = get_env_str("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
    # --- Sidebar ---
    with st.sidebar:
        st.header("⚙️ Управление")
        
        # Кнопка очистки чата
        if st.button("🗑️ Очистить чат", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # Кнопка пересборки индекса
        if st.button("🔄 Пересобрать индекс", use_container_width=True):
            with st.spinner("Пересборка индекса..."):
                # Импортируем здесь чтобы избежать circular import
                from rag_setup import rebuild_full_index
                
                success = rebuild_full_index(books_dir, index_dir)
                
                if success:
                    st.cache_resource.clear()
                    st.success("✅ Индекс пересобран!")
                    st.rerun()
                else:
                    st.error(f"❌ В папке {books_dir} нет PDF файлов.")
        
        st.divider()
        
        # Статус индекса
        st.subheader("📊 Статус индекса")
        config = load_index_config(index_dir)
        
        if config:
            st.success("✅ Индекс загружен")
            st.caption(f"**PDF:** {config.get('pdf_count', '?')}")
            st.caption(f"**Чанков:** {config.get('chunk_count', '?')}")
            st.caption(f"**Chunker:** {config.get('chunker', 'legacy')}")
            st.caption(f"**Max tokens:** {config.get('max_tokens', config.get('chunk_size', '?'))}")
            
            # Проверка несовпадения модели
            if check_embed_model_mismatch(config):
                st.warning("⚠️ EMBED_MODEL в env отличается от config.json")
        else:
            st.warning("⚠️ Индекс не найден")
            st.caption("Положите PDF в books/ и нажмите 'Пересобрать индекс'")
    
    # --- Проверка наличия индекса ---
    if not config:
        st.warning(
            "📂 **Индекс не найден.**\n\n"
            "1. Положите PDF файлы в папку `books/`\n"
            "2. Нажмите кнопку **Пересобрать индекс** в sidebar\n\n"
            "Или запустите в терминале: `python rag_setup.py`"
        )
        st.stop()
    
    # --- Загрузка ресурсов ---
    try:
        vectorstore = load_index(index_dir, config["embed_model"])
        if not vectorstore:
            st.error("❌ Не удалось загрузить индекс. Пересоберите его.")
            st.stop()
        
        retriever = get_retriever(vectorstore, top_k)
        llm = get_llm(ollama_model)
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {e}")
        st.stop()
    
    # --- Проверка Ollama ---
    # Проверяем при первом запуске или если флаг не установлен
    if "ollama_checked" not in st.session_state:
        with st.spinner("Проверка подключения к Ollama..."):
            if not check_ollama_connection(llm):
                st.error(
                    "❌ **Ollama недоступна.**\n\n"
                    "Как исправить:\n"
                    "1. Запустите Ollama: `ollama serve`\n"
                    f"2. Скачайте модель: `ollama pull {ollama_model}`"
                )
                st.stop()
            st.session_state.ollama_checked = True
    
    # --- Отображение истории чата ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Показываем источники если есть
            if message.get("sources"):
                with st.expander("📖 Источники"):
                    for source in message["sources"]:
                        st.caption(f"• {source}")
    
    # --- Ввод пользователя ---
    if user_input := st.chat_input("Задайте вопрос по книгам..."):
        # Добавляем сообщение пользователя
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Показываем сообщение пользователя
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Получаем ответ
        with st.chat_message("assistant"):
            with st.spinner("Думаю..."):
                try:
                    # Передаём историю БЕЗ текущего сообщения (оно уже в question)
                    history = st.session_state.messages[:-1]
                    answer, docs = ask_question(retriever, llm, user_input, history)
                    sources = format_sources(docs)
                    
                except RuntimeError as e:
                    answer = f"❌ Ошибка: {e}"
                    sources = []
                except Exception as e:
                    answer = f"❌ Произошла ошибка: {e}"
                    sources = []
            
            # Показываем ответ
            st.markdown(answer)
            
            # Показываем источники
            if sources:
                with st.expander("📖 Источники", expanded=True):
                    for source in sources:
                        st.caption(f"• {source}")
                
                # DEBUG: полный текст чанков
                with st.expander("🔍 DEBUG: Полный текст чанков", expanded=False):
                    for i, doc in enumerate(docs):
                        section = doc.metadata.get("section", doc.metadata.get("page_label", "?"))
                        st.markdown(f"**[{i+1}] {section}**")
                        st.code(doc.page_content, language=None)
                        st.divider()
            else:
                st.caption("📖 Источники: (нет)")
        
        # Сохраняем в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
