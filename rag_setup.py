"""
Сборка FAISS индекса: загрузка PDF, чанкинг, создание embeddings.

Stage 2 реализация согласно TECHNICAL_SPEC.md
"""

import glob
import json
import os
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker


# --- Константы по умолчанию ---
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BOOKS_DIR = "books/"
DEFAULT_INDEX_DIR = "rag_index/"
USE_SEMANTIC_CHUNKER = True  # Новый умный чанкинг


def is_e5_model(model_name: str) -> bool:
    """Проверяет, является ли модель E5 (требует префиксы query:/passage:)."""
    return "e5" in model_name.lower()


def add_passage_prefix(chunks: list, embed_model: str) -> list:
    """
    Добавляет 'passage: ' префикс к тексту чанков для E5 моделей.
    E5 обучался с этими префиксами — без них качество падает на 10-20%.
    """
    if not is_e5_model(embed_model):
        return chunks
    
    print(f"  📝 Добавляем 'passage:' префикс для E5 модели")
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
    return chunks


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


def load_documents(pdf_paths: list[str]) -> list:
    """
    Загружает PDF постранично через PyPDFLoader(mode="page").
    Каждый Document содержит metadata: {source, page}.
    """
    all_pages = []
    for pdf_path in pdf_paths:
        print(f"  📄 Загрузка: {pdf_path}")
        loader = PyPDFLoader(pdf_path, mode="page")
        pages = loader.load()
        all_pages.extend(pages)
    return all_pages


def split_documents(docs: list, chunk_size: int, chunk_overlap: int, embed_model: str = None) -> list:
    """
    Разбивает документы на чанки.
    
    Если USE_SEMANTIC_CHUNKER=True:
      Использует SemanticChunker — группирует по смыслу через embeddings.
      Связанные предложения остаются вместе, таблицы не разрываются.
    
    Иначе:
      RecursiveCharacterTextSplitter — режет по количеству символов.
    """
    if USE_SEMANTIC_CHUNKER and embed_model:
        print(f"  🧠 Используем SemanticChunker (умный чанкинг по смыслу)")
        embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",  # или "standard_deviation"
            breakpoint_threshold_amount=70,  # чем выше, тем крупнее чанки
        )
        # SemanticChunker работает с текстом, нужно обработать каждый документ
        all_chunks = []
        for doc in docs:
            chunks = splitter.create_documents(
                [doc.page_content],
                metadatas=[doc.metadata]
            )
            all_chunks.extend(chunks)
        return all_chunks
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
            # Добавляем page_label если есть
            if "page_label" in chunk.metadata:
                record["page_label"] = chunk.metadata["page_label"]
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
    }
    config_path = os.path.join(index_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  📝 Конфиг сохранён: {config_path}")


def build_index(chunks: list, index_dir: str, embed_model: str) -> None:
    """Создаёт embeddings и сохраняет FAISS индекс."""
    # Добавляем passage: префикс для E5 моделей
    chunks = add_passage_prefix(chunks, embed_model)
    
    print(f"  🔢 Создание embeddings ({embed_model})...")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    
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
    print("🚀 Сборка FAISS индекса")
    print("=" * 50)
    print(f"  📁 Папка PDF: {books_dir}")
    print(f"  📁 Папка индекса: {index_dir}")
    print(f"  📏 chunk_size: {chunk_size}")
    print(f"  📏 chunk_overlap: {chunk_overlap}")
    print(f"  🧠 embed_model: {embed_model}")
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

    # 2. Загружаем страницы
    print("📖 Загрузка страниц...")
    pages = load_documents(pdf_files)
    print(f"   Загружено страниц: {len(pages)}")
    print()

    # 3. Чанкинг
    print("✂️ Разбиение на чанки...")
    chunks = split_documents(pages, chunk_size, chunk_overlap, embed_model)
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
