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

# Системный промпт — оптимизирован для полных ответов
SYSTEM_PROMPT = """Ты — эксперт-консультант, готовящий материал для обучающего курса.

ЯЗЫК: Отвечай ТОЛЬКО на русском языке. Никакого китайского, английского или других языков.

ГЛАВНОЕ ПРАВИЛО: Отвечай ТОЛЬКО на основе контекста ниже.
Если информации нет — скажи "в предоставленных материалах это не описано".

ТРЕБОВАНИЯ К ОТВЕТУ:
1. Минимум 3-5 абзацев — краткие ответы недопустимы
2. Перед ответом мысленно проанализируй ВСЕ фрагменты контекста
3. Структура: определение → подробное объяснение → примеры → итог
4. Раскрой каждый пункт: что это, зачем нужно, как работает
5. Приводи цитаты и формулы дословно из контекста
6. Если есть связанные понятия — объясни связь между ними
7. Пиши простым языком, как для неспециалиста

ФОРМАТ:
- Термины выделяй **жирным**
- Для перечислений используй нумерацию
- Формулы: ROE = Чистая прибыль / Собственный капитал

Контекст ({chunk_count} фрагментов):
{context}

Вопрос: {question}

Развёрнутый ответ (на русском):"""


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
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
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
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


# --- Контекст и промпт ---

def format_context(docs: list) -> str:
    """
    Собирает контекст из документов с нумерацией "X из Y".
    Порядок: менее релевантные сначала, самый релевантный — в конце.
    (LLM лучше запоминают конец контекста — "recency bias")
    """
    if not docs:
        return ""
    
    total = len(docs)
    reversed_docs = list(reversed(docs))
    
    fragments = []
    for i, doc in enumerate(reversed_docs, 1):
        section = doc.metadata.get("section", "?")
        header = f"[Фрагмент {i} из {total}, {section}]"
        fragments.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(fragments)


def build_prompt(context: str, question: str, chunk_count: int) -> str:
    """Собирает финальный промпт для LLM."""
    return SYSTEM_PROMPT.format(
        context=context, 
        question=question,
        chunk_count=chunk_count
    )


# --- Форматирование источников ---

def format_sources(docs: list) -> list[str]:
    """
    Форматирует и дедуплицирует список источников.
    Сохраняет порядок первого появления.
    """
    seen = set()
    result = []
    for doc in docs:
        filename = os.path.basename(doc.metadata.get("source", "unknown"))
        section = doc.metadata.get("section", "")
        source = f"{filename} [{section}]" if section else filename
        if source not in seen:
            seen.add(source)
            result.append(source)
    return result



# --- Главная функция ответа ---

def ask_question(retriever, llm, question: str) -> tuple[str, list]:
    """
    Отвечает на вопрос используя retriever и llm напрямую.
    Возвращает (answer, docs).
    """
    # 1. Получаем документы
    docs = retriever.invoke(question)
    
    # 2. Если нет документов — НЕ вызываем LLM
    if not docs:
        return "В книгах нет информации по этому вопросу.", []
    
    # 3. Формируем контекст и получаем ответ
    context = format_context(docs)
    prompt = build_prompt(context, question, len(docs))
    answer = llm.invoke(prompt)
    
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
    index_dir = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
    books_dir = os.getenv("BOOKS_DIR", "books/")
    top_k = get_env_int("TOP_K", DEFAULT_TOP_K)
    ollama_model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
    
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
                    answer, docs = ask_question(retriever, llm, user_input)
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
        
        # Сохраняем в историю
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
