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
DEFAULT_TOP_K = 8  # Увеличено с 4 для лучшего покрытия контекста
DEFAULT_OLLAMA_MODEL = "llama3"
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Системный промпт (из TECHNICAL_SPEC.md)
SYSTEM_PROMPT = """Ты — помощник, отвечающий на основе предоставленного контекста из книг.

ПРАВИЛА:
1. Отвечай на основе информации из контекста. Если вопрос абстрактный (например, "какие виды X есть"), ищи в контексте конкретные примеры X и перечисляй их.
2. Отвечай ТОЛЬКО на РУССКОМ языке, даже если контекст содержит английские термины.
3. Давай ПОДРОБНЫЕ и РАЗВЁРНУТЫЕ ответы. Если в контексте есть списки или перечисления — приводи их ПОЛНОСТЬЮ.
4. Если в контексте НЕТ НИКАКОЙ связанной информации — отвечай: "В книгах нет информации по этому вопросу." Но если есть СВЯЗАННЫЕ термины или примеры — используй их для ответа.
5. НЕ придумывай факты, которых нет в контексте.
6. НЕ добавляй источники/ссылки/страницы в текст ответа. Источники будут показаны отдельно автоматически.
7. НЕ придумывай номера страниц или названия файлов.
8. Используй форматирование: списки, абзацы для читабельности.

Контекст из книг:
{context}

Вопрос: {question}

Ответ (на русском языке):"""


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
    
    # Проверяем обязательные ключи
    required = ["embed_model", "chunk_size", "chunk_overlap"]
    for key in required:
        if key not in config:
            raise RuntimeError(
                f"❌ В config.json отсутствует ключ '{key}'.\n"
                f"   Пересоберите индекс: python rag_setup.py"
            )
    return config


def get_embeddings(config: dict) -> HuggingFaceEmbeddings:
    """
    Создаёт embeddings используя embed_model ИЗ CONFIG (не из env!).
    Это гарантирует совпадение модели при сборке и при query.
    """
    model_name = config["embed_model"]
    return HuggingFaceEmbeddings(model_name=model_name)


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
    """Собирает контекст из документов."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def build_prompt(context: str, question: str) -> str:
    """Собирает финальный промпт для LLM."""
    return SYSTEM_PROMPT.format(context=context, question=question)


# --- Форматирование источников ---

def format_source(doc) -> str:
    """
    Форматирует источник в ЕДИНЫЙ формат: "book.pdf [стр. 23]"
    """
    filename = os.path.basename(doc.metadata.get("source", "unknown"))
    
    # Предпочитаем page_label, иначе page+1
    page_label = doc.metadata.get("page_label")
    if page_label:
        page = page_label
    else:
        page_num = doc.metadata.get("page")
        page = str(page_num + 1) if page_num is not None else "?"
    
    return f"{filename} [стр. {page}]"


def format_sources(docs: list) -> list[str]:
    """
    Форматирует и дедуплицирует список источников.
    Сохраняет порядок первого появления.
    """
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
    
    # 2. Получаем документы
    # Основной метод — invoke, fallback — get_relevant_documents
    try:
        docs = retriever.invoke(full_question)
    except AttributeError:
        docs = retriever.get_relevant_documents(full_question)
    
    # Проверяем что получили список
    if not isinstance(docs, list):
        docs = list(docs) if docs else []
    
    # 3. Если нет документов — НЕ вызываем LLM
    if not docs:
        return "В книгах нет информации по этому вопросу.", []
    
    # 4. Собираем контекст
    context = format_context(docs)
    
    # 5. Собираем промпт
    prompt = build_prompt(context, full_question)
    
    # 6. Вызываем LLM
    answer = llm.invoke(prompt)
    
    # 7. Возвращаем ответ и документы
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
            st.caption(f"**chunk_size:** {config.get('chunk_size', '?')}")
            st.caption(f"**overlap:** {config.get('chunk_overlap', '?')}")
            
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
