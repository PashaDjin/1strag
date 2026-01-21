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
DEFAULT_RERANK_TOP_K = 4  # После reranking оставляем лучшие 4
DEFAULT_OLLAMA_MODEL = "qwen2.5:14b"  # Лучший для русского. Альтернатива: qwen2.5:7b
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ENABLE_RERANKING = False  # Отключено: лучше анализировать ВСЕ чанки

# Системный промпт с Chain-of-Thought
SYSTEM_PROMPT = """Ты — эксперт-аналитик, отвечающий на вопросы на основе книг.

МЕТОД ОТВЕТА (выполняй последовательно):

ШАГ 1 — ИЗВЛЕЧЕНИЕ: Найди в контексте ВСЕ элементы, связанные с вопросом:
- Термины и определения
- Принципы и правила
- Формулы и методы расчёта
- Примеры и кейсы
- Числа и факты

ШАГ 2 — АНАЛИЗ: Для каждого найденного элемента определи, как он отвечает на вопрос.

ШАГ 3 — СИНТЕЗ: Объедини всё в структурированный полный ответ.

ПРАВИЛА:
- Отвечай ТОЛЬКО на РУССКОМ языке
- Анализируй ВСЕ фрагменты контекста — информация распределена по ним
- Ищи таблицы (строки вида "Название ... Число") и извлекай ВСЕ строки
- НЕ придумывай факты, которых нет в контексте
- НЕ добавляй источники в текст — они покажутся автоматически
- Если информации нет — скажи "В книгах нет информации по этому вопросу"

Контекст из книг:
{context}

Вопрос: {question}

Ответ:"""


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
    """
    Собирает контекст из документов с нумерацией "X из Y".
    Порядок: менее релевантные сначала, самый релевантный — в конце.
    (LLM лучше запоминают конец контекста — "recency bias")
    """
    if not docs:
        return ""
    
    total = len(docs)
    # Переворачиваем: самый релевантный будет последним
    reversed_docs = list(reversed(docs))
    
    fragments = []
    for i, doc in enumerate(reversed_docs, 1):
        # Получаем номер страницы
        page_label = doc.metadata.get("page_label")
        if not page_label:
            page_num = doc.metadata.get("page")
            page_label = str(page_num + 1) if page_num is not None else "?"
        
        # "X из Y" создаёт ощущение чек-листа для LLM
        header = f"[Фрагмент {i} из {total}, стр. {page_label}]"
        fragments.append(f"{header}\n{doc.page_content}")
    
    return "\n\n---\n\n".join(fragments)


def build_prompt(context: str, question: str) -> str:
    """Собирает финальный промпт для LLM."""
    return SYSTEM_PROMPT.format(context=context, question=question)


# --- LLM-based Reranking ---

RERANK_PROMPT = """Оцени релевантность текста к вопросу по шкале 0-10.
Отвечай ТОЛЬКО числом от 0 до 10.

Вопрос: {question}

Текст: {chunk}

Оценка (0-10):"""


def rerank_docs(docs: list, question: str, llm, top_k: int = 4) -> list:
    """
    LLM-based reranking: оценивает релевантность каждого чанка.
    Возвращает top_k лучших документов отсортированных по релевантности.
    
    Это двухэтапный процесс:
    1. Retriever возвращает много чанков (широкий охват)
    2. LLM оценивает каждый и отбирает самые релевантные (точность)
    """
    if not docs or len(docs) <= top_k:
        return docs
    
    scored_docs = []
    
    for doc in docs:
        # Ограничиваем размер чанка для оценки (экономим токены)
        chunk_preview = doc.page_content[:800]
        prompt = RERANK_PROMPT.format(question=question, chunk=chunk_preview)
        
        try:
            response = llm.invoke(prompt)
            # Извлекаем число из ответа
            score = extract_score(response)
            scored_docs.append((score, doc))
        except Exception:
            # При ошибке даём средний балл
            scored_docs.append((5, doc))
    
    # Сортируем по убыванию оценки
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Возвращаем top_k лучших
    return [doc for score, doc in scored_docs[:top_k]]


def extract_score(response: str) -> int:
    """Извлекает числовую оценку из ответа LLM."""
    import re
    # Ищем первое число в ответе
    match = re.search(r'\b(\d+)\b', response.strip())
    if match:
        score = int(match.group(1))
        return min(max(score, 0), 10)  # Ограничиваем 0-10
    return 5  # Default


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
    
    # 2. Получаем документы (широкий охват)
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
    
    # 3.5 Reranking: LLM отбирает самые релевантные чанки
    if ENABLE_RERANKING and len(docs) > DEFAULT_RERANK_TOP_K:
        docs = rerank_docs(
            docs=docs,
            question=full_question,
            llm=llm,
            top_k=DEFAULT_RERANK_TOP_K
        )
    
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
