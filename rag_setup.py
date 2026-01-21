"""
Сборка FAISS индекса: загрузка PDF через Docling, чанкинг, создание embeddings.

Docling конвертирует PDF в Markdown с сохранением таблиц!
"""

import glob
import json
import os
from datetime import datetime

from docling.document_converter import DocumentConverter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# --- Константы по умолчанию ---
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BOOKS_DIR = "books/"
DEFAULT_INDEX_DIR = "rag_index/"

# Docling конвертирует таблицы в Markdown — используем MarkdownTextSplitter
USE_MARKDOWN_SPLITTER = True


def is_e5_model(model_name: str) -> bool:
    """Проверяет, является ли модель E5 (требует префиксы query:/passage:)."""
    return "e5" in model_name.lower()


class E5Embeddings:
    """
    Wrapper для HuggingFaceEmbeddings с автоматическими E5 префиксами.
    
    E5 модели требуют:
    - 'passage: ' для документов при индексации
    - 'query: ' для запросов при поиске
    
    Этот wrapper добавляет префиксы автоматически, НЕ модифицируя оригинальный текст.
    Текст в page_content остаётся чистым!
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base = HuggingFaceEmbeddings(model_name=model_name)
        self._is_e5 = is_e5_model(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed документов с 'passage:' префиксом."""
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.base.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
        """Embed запроса с 'query:' префиксом."""
        if self._is_e5:
            text = f"query: {text}"
        return self.base.embed_query(text)


def get_env_int(name: str, default: int) -> int:
    """Безопасно читает int из env."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"⚠️ Неверное значение {name}={value}, используется {default}")
        return default


def get_pdf_files(books_dir: str) -> list[str]:
    """Возвращает список путей к PDF в папке books/."""
    pattern = os.path.join(books_dir, "*.pdf")
    return sorted(glob.glob(pattern))


def load_documents_with_docling(pdf_paths: list[str]) -> list[Document]:
    """
    Загружает PDF через Docling и конвертирует в Markdown.
    
    Docling:
    - Распознаёт структуру документа (заголовки, списки)
    - Конвертирует таблицы в Markdown таблицы
    - Сохраняет форматирование
    
    Возвращает список LangChain Document с page_content в Markdown.
    """
    converter = DocumentConverter()
    all_docs = []
    
    for pdf_path in pdf_paths:
        print(f"  📄 Docling конвертирует: {pdf_path}")
        
        try:
            # Конвертируем PDF в структурированный документ
            result = converter.convert(pdf_path)
            
            # Экспортируем в Markdown (таблицы станут Markdown таблицами!)
            markdown_content = result.document.export_to_markdown()
            
            # Создаём LangChain Document
            doc = Document(
                page_content=markdown_content,
                metadata={
                    "source": pdf_path,
                    "page": 0,  # Docling не даёт постраничную разбивку
                    "format": "markdown",
                    "converter": "docling",
                }
            )
            all_docs.append(doc)
            
            print(f"    ✅ Сконвертировано: {len(markdown_content)} символов")
            
        except Exception as e:
            print(f"    ❌ Ошибка конвертации {pdf_path}: {e}")
            continue
    
    return all_docs


def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Разбивает документы на чанки.
    
    Для Markdown (Docling) используем MarkdownTextSplitter:
    - Режет по заголовкам (# ## ###)
    - Сохраняет структуру таблиц
    
    Для обычного текста — RecursiveCharacterTextSplitter.
    """
    if USE_MARKDOWN_SPLITTER:
        print(f"  📝 Используем MarkdownTextSplitter (для Docling)")
        splitter = MarkdownTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        print(f"  ✂️ Используем RecursiveCharacterTextSplitter")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )
    
    return splitter.split_documents(docs)


def save_chunks_for_debug(chunks: list, path: str) -> None:
    """
    Сохраняет чанки в JSONL для отладки.
    ВНИМАНИЕ: Файл может содержать текст из PDF (копирайт). 
    Не коммитить в git!
    """
    with open(path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {
                "chunk_id": i,
                "source": chunk.metadata.get("source", "unknown"),
                "page": chunk.metadata.get("page", 0),
                "text": chunk.page_content,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_index_config(
    index_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    embed_model: str,
    pdf_files: list[str],
    chunk_count: int,
) -> None:
    """Сохраняет config.json с параметрами сборки индекса."""
    config = {
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "built_at": datetime.now().isoformat(),
        "pdf_count": len(pdf_files),
        "pdf_files": [os.path.basename(p) for p in pdf_files],
        "chunk_count": chunk_count,
        "pdf_parser": "docling",
        "markdown_splitter": USE_MARKDOWN_SPLITTER,
    }
    config_path = os.path.join(index_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  📝 Конфиг сохранён: {config_path}")


def build_index(chunks: list, index_dir: str, embed_model: str) -> None:
    """
    Создаёт embeddings и сохраняет FAISS индекс.
    
    ВАЖНО: Используем E5Embeddings wrapper который добавляет prefix
    только при создании embedding, НЕ модифицируя page_content!
    """
    print(f"  🔢 Создание embeddings ({embed_model})...")
    
    # E5Embeddings добавляет 'passage:' автоматически при embed_documents
    embeddings = E5Embeddings(model_name=embed_model)
    
    print("  📦 Построение FAISS индекса...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Создаём папку если не существует
    os.makedirs(index_dir, exist_ok=True)
    
    vectorstore.save_local(index_dir)
    print(f"  💾 Индекс сохранён: {index_dir}")


def rebuild_full_index(books_dir: str, index_dir: str) -> bool:
    """
    Главная функция: вызывает все вышеперечисленные.
    Возвращает True если успешно, False если нет PDF.
    """
    # Читаем параметры из env
    chunk_size = get_env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = get_env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    debug_dump = os.getenv("DEBUG_DUMP_CHUNKS", "0") == "1"

    print("=" * 50)
    print("🚀 Сборка FAISS индекса (Docling + Markdown)")
    print("=" * 50)
    print(f"  📁 Папка PDF: {books_dir}")
    print(f"  📁 Папка индекса: {index_dir}")
    print(f"  📏 chunk_size: {chunk_size}")
    print(f"  📏 chunk_overlap: {chunk_overlap}")
    print(f"  🧠 embed_model: {embed_model}")
    print(f"  📄 PDF парсер: Docling")
    print()

    # 1. Получаем список PDF
    pdf_files = get_pdf_files(books_dir)
    if not pdf_files:
        print(f"❌ В папке {books_dir} нет PDF файлов.")
        print("   Положите PDF файлы в папку и запустите снова.")
        return False

    print(f"📚 Найдено PDF: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"   • {pdf}")
    print()

    # 2. Загружаем через Docling (конвертация в Markdown)
    print("📖 Конвертация PDF через Docling...")
    docs = load_documents_with_docling(pdf_files)
    if not docs:
        print("❌ Не удалось загрузить ни одного документа")
        return False
    print(f"   Загружено документов: {len(docs)}")
    print()

    # 3. Чанкинг
    print("✂️ Разбиение на чанки...")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"   Создано чанков: {len(chunks)}")
    print()

    # 4. Опционально: дамп чанков
    if debug_dump:
        chunks_path = "chunks.jsonl"
        print(f"🔍 DEBUG: Сохранение чанков в {chunks_path}...")
        save_chunks_for_debug(chunks, chunks_path)
        print(f"   ⚠️ ВНИМАНИЕ: файл может содержать копирайтный текст!")
        print()

    # 5. Строим и сохраняем индекс
    print("🔨 Построение индекса...")
    build_index(chunks, index_dir, embed_model)
    print()

    # 6. Сохраняем конфиг
    save_index_config(
        index_dir=index_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model=embed_model,
        pdf_files=pdf_files,
        chunk_count=len(chunks),
    )

    print()
    print("=" * 50)
    print("✅ Индекс успешно создан!")
    print("=" * 50)
    return True


def main():
    """Точка входа CLI."""
    books_dir = os.getenv("BOOKS_DIR", DEFAULT_BOOKS_DIR)
    index_dir = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
    
    rebuild_full_index(books_dir, index_dir)


if __name__ == "__main__":
    main()
