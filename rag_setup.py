"""
Сборка FAISS индекса: Docling + HybridChunker.

HybridChunker:
- НЕ разрывает таблицы
- Понимает структуру документа (заголовки → чанки)
- Token-aware (не режет посередине)
- Добавляет headings в metadata
- Таблицы в Markdown формате (MarkdownTableSerializer)
"""

import glob
import json
import os
from datetime import datetime
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# --- Константы по умолчанию ---
DEFAULT_MAX_TOKENS = 500  # Токены! E5-base лимит 512, оставляем запас
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BOOKS_DIR = "books/"
DEFAULT_INDEX_DIR = "rag_index/"
CACHE_DIR = ".docling_cache"  # Кеш DoclingDocument в JSON


class MarkdownTableSerializerProvider(ChunkingSerializerProvider):
    """
    Кастомный provider для HybridChunker.
    Таблицы сериализуются в Markdown формат вместо triplet (Key=Value).
    """
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


def is_e5_model(model_name: str) -> bool:
    """Проверяет, является ли модель E5 (требует префиксы query:/passage:)."""
    return "e5" in model_name.lower()


class E5Embeddings:
    """
    Wrapper для HuggingFaceEmbeddings с автоматическими E5 префиксами.
    
    E5 модели требуют:
    - 'passage: ' для документов при индексации
    - 'query: ' для запросов при поиске
    
    Этот wrapper добавляет префиксы автоматически, НЕ модифицируя page_content.
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


def get_cache_path(pdf_path: str) -> str:
    """Возвращает путь к JSON кешу для PDF."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    pdf_name = Path(pdf_path).stem
    return os.path.join(CACHE_DIR, f"{pdf_name}.json")


def load_docling_document(pdf_path: str, converter: DocumentConverter):
    """
    Загружает DoclingDocument из кеша или конвертирует PDF.
    
    Кеширование экономит ~2-3 минуты при повторных сборках.
    """
    cache_path = get_cache_path(pdf_path)
    
    # Проверяем кеш
    if os.path.exists(cache_path):
        # Проверяем что PDF не изменился
        pdf_mtime = os.path.getmtime(pdf_path)
        cache_mtime = os.path.getmtime(cache_path)
        
        if cache_mtime > pdf_mtime:
            print(f"    📦 Загружаем из кеша: {cache_path}")
            try:
                from docling_core.types import DoclingDocument as DoclingDoc
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return DoclingDoc.model_validate(data)
            except Exception as e:
                print(f"    ⚠️ Ошибка загрузки кеша: {e}, конвертируем заново")
    
    # Конвертируем PDF
    print(f"    🔄 Конвертируем PDF...")
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Сохраняем в кеш
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(doc.export_to_dict(), f, ensure_ascii=False)
        print(f"    💾 Сохранено в кеш: {cache_path}")
    except Exception as e:
        print(f"    ⚠️ Не удалось сохранить кеш: {e}")
    
    return doc


def chunk_with_hybrid_chunker(
    docling_doc,
    pdf_path: str,
    max_tokens: int,
) -> list[Document]:
    """
    Чанкит DoclingDocument через HybridChunker.
    
    HybridChunker:
    - Понимает структуру документа
    - НЕ разрывает таблицы посередине
    - Token-aware (учитывает max_tokens)
    - Добавляет headings в metadata
    - Таблицы в Markdown формате (не Key=Value каша)
    
    contextualize() добавляет заголовки в текст для лучшего embedding.
    """
    chunker = HybridChunker(
        max_tokens=max_tokens,
        merge_peers=True,  # Объединяет маленькие соседние чанки
        serializer_provider=MarkdownTableSerializerProvider(),  # Таблицы в Markdown!
    )
    
    chunks = list(chunker.chunk(dl_doc=docling_doc))
    
    # Конвертируем в LangChain Documents
    documents = []
    for i, chunk in enumerate(chunks):
        # Получаем headings для section
        headings = []
        if hasattr(chunk, 'meta') and chunk.meta:
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                headings = chunk.meta.headings
        
        section = " > ".join(headings) if headings else f"Чанк {i+1}"
        
        # contextualize() добавляет headings в текст для лучшего embedding
        # Это помогает E5 понять контекст чанка
        try:
            enriched_text = chunker.contextualize(chunk)
        except Exception:
            enriched_text = chunk.text
        
        doc = Document(
            page_content=enriched_text,
            metadata={
                "source": pdf_path,
                "section": section,
                "headings": headings,
                "chunk_id": i,
            }
        )
        documents.append(doc)
    
    return documents


def save_chunks_for_debug(chunks: list[Document], path: str) -> None:
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
                "section": chunk.metadata.get("section", ""),
                "headings": chunk.metadata.get("headings", []),
                "text": chunk.page_content,
                "text_len": len(chunk.page_content),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_index_config(
    index_dir: str,
    max_tokens: int,
    embed_model: str,
    pdf_files: list[str],
    chunk_count: int,
) -> None:
    """Сохраняет config.json с параметрами сборки индекса."""
    config = {
        "embed_model": embed_model,
        "chunker": "HybridChunker",
        "max_tokens": max_tokens,
        "built_at": datetime.now().isoformat(),
        "pdf_count": len(pdf_files),
        "pdf_files": [os.path.basename(p) for p in pdf_files],
        "chunk_count": chunk_count,
    }
    config_path = os.path.join(index_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  📝 Конфиг сохранён: {config_path}")


def build_index(chunks: list[Document], index_dir: str, embed_model: str) -> None:
    """Создаёт embeddings и сохраняет FAISS индекс."""
    print(f"  🔢 Создание embeddings ({embed_model})...")
    
    # Используем E5Embeddings wrapper — он сам добавляет prefix при embed
    embeddings = E5Embeddings(model_name=embed_model)
    
    print("  📦 Построение FAISS индекса...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Создаём папку если не существует
    os.makedirs(index_dir, exist_ok=True)
    
    vectorstore.save_local(index_dir)
    print(f"  💾 Индекс сохранён: {index_dir}")


def rebuild_full_index(books_dir: str, index_dir: str) -> bool:
    """
    Главная функция сборки индекса.
    
    Pipeline:
    1. Найти PDF файлы
    2. Конвертировать через Docling (с кешированием)
    3. Чанкинг через HybridChunker
    4. Создать embeddings (E5 с prefix)
    5. Сохранить FAISS индекс
    """
    # Читаем параметры из env
    max_tokens = get_env_int("MAX_TOKENS", DEFAULT_MAX_TOKENS)
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    debug_dump = os.getenv("DEBUG_DUMP_CHUNKS", "0") == "1"

    print("=" * 50)
    print("🚀 Сборка FAISS индекса (Docling + HybridChunker)")
    print("=" * 50)
    print(f"  📁 Папка PDF: {books_dir}")
    print(f"  📁 Папка индекса: {index_dir}")
    print(f"  📏 max_tokens: {max_tokens}")
    print(f"  🧠 embed_model: {embed_model}")
    print(f"  🔧 chunker: HybridChunker")
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

    # 2. Конвертируем и чанкаем
    converter = DocumentConverter()
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"📄 Обработка: {pdf_path}")
        
        try:
            # Загружаем DoclingDocument (из кеша или конвертируем)
            docling_doc = load_docling_document(pdf_path, converter)
            
            # Чанкинг через HybridChunker
            chunks = chunk_with_hybrid_chunker(docling_doc, pdf_path, max_tokens)
            all_chunks.extend(chunks)
            
            print(f"    ✅ Создано чанков: {len(chunks)}")
            
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            continue
    
    print()
    print(f"📊 Всего чанков: {len(all_chunks)}")
    print()

    if not all_chunks:
        print("❌ Не удалось создать ни одного чанка")
        return False

    # 3. Опционально: дамп чанков для отладки
    if debug_dump:
        chunks_path = "chunks.jsonl"
        print(f"🔍 DEBUG: Сохранение чанков в {chunks_path}...")
        save_chunks_for_debug(all_chunks, chunks_path)
        print(f"   ⚠️ ВНИМАНИЕ: файл может содержать копирайтный текст!")
        print()

    # 4. Строим и сохраняем индекс
    print("🔨 Построение индекса...")
    build_index(all_chunks, index_dir, embed_model)
    print()

    # 5. Сохраняем конфиг
    save_index_config(
        index_dir=index_dir,
        max_tokens=max_tokens,
        embed_model=embed_model,
        pdf_files=pdf_files,
        chunk_count=len(all_chunks),
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
