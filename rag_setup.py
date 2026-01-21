"""
Сборка FAISS индекса с Docling + HybridChunker.

Улучшения:
- HybridChunker: не разрывает таблицы, token-aware
- Picture description: описания схем и графиков
- Headings: контекст заголовков в метаданных
- JSON cache: кеширование DoclingDocument
- contextualize(): заголовки добавляются в embedding
"""

import glob
import json
import os
from datetime import datetime
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


# --- Константы ---
DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BOOKS_DIR = "books/"
DEFAULT_INDEX_DIR = "rag_index/"
DEFAULT_CACHE_DIR = "docling_cache/"
DEFAULT_MAX_TOKENS = 400  # Для E5 (max 512, оставляем запас)

# Picture description — опционально, требует VLM модель
ENABLE_PICTURE_DESCRIPTION = False  # Включи если нужны описания схем


def is_e5_model(model_name: str) -> bool:
    """Проверяет, является ли модель E5."""
    return "e5" in model_name.lower()


class E5Embeddings:
    """
    Wrapper для HuggingFaceEmbeddings с E5 префиксами.
    
    Добавляет префиксы только при embed, НЕ модифицируя текст.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base = HuggingFaceEmbeddings(model_name=model_name)
        self._is_e5 = is_e5_model(model_name)
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        return self.base.embed_documents(texts)
    
    def embed_query(self, text: str) -> list[float]:
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
        return default


def get_pdf_files(books_dir: str) -> list[str]:
    """Возвращает список PDF файлов."""
    pattern = os.path.join(books_dir, "*.pdf")
    return sorted(glob.glob(pattern))


def get_cache_path(pdf_path: str, cache_dir: str) -> str:
    """Возвращает путь к JSON кешу для PDF."""
    pdf_name = Path(pdf_path).stem
    return os.path.join(cache_dir, f"{pdf_name}.json")


def load_cached_document(cache_path: str):
    """Загружает DoclingDocument из JSON кеша."""
    from docling_core.types.doc.document import DoclingDocument as DLDocument
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return DLDocument.model_validate(data)
    except Exception as e:
        print(f"    ⚠️ Ошибка загрузки кеша: {e}")
        return None


def save_document_cache(doc, cache_path: str) -> None:
    """Сохраняет DoclingDocument в JSON кеш."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, ensure_ascii=False, indent=2)


def create_converter() -> DocumentConverter:
    """
    Создаёт DocumentConverter с оптимальными настройками.
    """
    pipeline_options = PdfPipelineOptions()
    
    # OCR отключаем — книга Герасименко не сканирована
    pipeline_options.do_ocr = False
    
    # Picture description — описания схем
    if ENABLE_PICTURE_DESCRIPTION:
        try:
            from docling.datamodel.pipeline_options import smolvlm_picture_description
            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = smolvlm_picture_description
            print("  🖼️ Picture description включено (SmolVLM)")
        except ImportError:
            print("  ⚠️ SmolVLM не установлен, picture description отключено")
            pipeline_options.do_picture_description = False
    else:
        pipeline_options.do_picture_description = False
    
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf_to_docling(pdf_path: str, converter: DocumentConverter, cache_dir: str):
    """
    Конвертирует PDF в DoclingDocument.
    Использует кеш если есть.
    """
    cache_path = get_cache_path(pdf_path, cache_dir)
    
    # Проверяем кеш
    cached_doc = load_cached_document(cache_path)
    if cached_doc is not None:
        print(f"  📦 Загружено из кеша: {cache_path}")
        return cached_doc
    
    # Конвертируем
    print(f"  📄 Конвертация: {pdf_path}")
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Сохраняем в кеш
    save_document_cache(doc, cache_path)
    print(f"  💾 Сохранено в кеш: {cache_path}")
    
    return doc


def create_chunker(embed_model: str, max_tokens: int) -> HybridChunker:
    """
    Создаёт HybridChunker с токенизатором embedding модели.
    """
    print(f"  🔧 Создание HybridChunker (max_tokens={max_tokens}, merge_peers=False)")
    
    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    
    return HybridChunker(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        merge_peers=False,  # Много маленьких чанков — лучше для retrieval
    )


def chunk_document(doc, chunker: HybridChunker, source_name: str) -> list[Document]:
    """
    Разбивает DoclingDocument на LangChain Documents.
    
    Использует contextualize() для добавления headings в текст.
    Сохраняет headings в metadata.
    """
    chunks = list(chunker.chunk(dl_doc=doc))
    
    documents = []
    for idx, chunk in enumerate(chunks):
        # contextualize добавляет заголовки к тексту для embedding
        contextualized_text = chunker.contextualize(chunk)
        
        # Извлекаем headings из метаданных
        headings = []
        if hasattr(chunk, 'meta') and chunk.meta:
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                headings = chunk.meta.headings
        
        # Создаём LangChain Document
        doc = Document(
            page_content=contextualized_text,
            metadata={
                "source": source_name,
                "chunk_id": idx,
                "headings": headings,
                # Первый заголовок как "section" для отображения
                "section": headings[0] if headings else "",
            }
        )
        documents.append(doc)
    
    return documents


def save_chunks_for_debug(documents: list[Document], path: str) -> None:
    """Сохраняет чанки в JSONL для отладки."""
    with open(path, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "source": doc.metadata.get("source", "unknown"),
                "section": doc.metadata.get("section", ""),
                "headings": doc.metadata.get("headings", []),
                "text": doc.page_content,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_index_config(
    index_dir: str,
    embed_model: str,
    max_tokens: int,
    pdf_files: list[str],
    chunk_count: int,
) -> None:
    """Сохраняет config.json."""
    config = {
        "embed_model": embed_model,
        "max_tokens": max_tokens,
        "merge_peers": False,
        "built_at": datetime.now().isoformat(),
        "pdf_count": len(pdf_files),
        "pdf_files": [os.path.basename(p) for p in pdf_files],
        "chunk_count": chunk_count,
        "chunker": "HybridChunker",
        "parser": "Docling",
        "picture_description": ENABLE_PICTURE_DESCRIPTION,
    }
    
    config_path = os.path.join(index_dir, "config.json")
    os.makedirs(index_dir, exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  📝 Конфиг: {config_path}")


def build_index(documents: list[Document], index_dir: str, embed_model: str) -> None:
    """Создаёт FAISS индекс."""
    print(f"  🔢 Создание embeddings ({embed_model})...")
    
    embeddings = E5Embeddings(model_name=embed_model)
    
    print("  📦 Построение FAISS индекса...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"  💾 Индекс сохранён: {index_dir}")


def rebuild_full_index(books_dir: str, index_dir: str) -> bool:
    """
    Главная функция сборки индекса.
    """
    # Параметры из env
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    cache_dir = os.getenv("CACHE_DIR", DEFAULT_CACHE_DIR)
    max_tokens = get_env_int("MAX_TOKENS", DEFAULT_MAX_TOKENS)
    debug_dump = os.getenv("DEBUG_DUMP_CHUNKS", "0") == "1"

    print("=" * 60)
    print("🚀 Сборка FAISS индекса (Docling + HybridChunker)")
    print("=" * 60)
    print(f"  📁 PDF папка: {books_dir}")
    print(f"  📁 Индекс: {index_dir}")
    print(f"  📁 Кеш: {cache_dir}")
    print(f"  🧠 Embed model: {embed_model}")
    print(f"  📏 Max tokens: {max_tokens}")
    print(f"  🖼️ Picture description: {ENABLE_PICTURE_DESCRIPTION}")
    print()

    # 1. Получаем PDF файлы
    pdf_files = get_pdf_files(books_dir)
    if not pdf_files:
        print(f"❌ В папке {books_dir} нет PDF файлов.")
        return False

    print(f"📚 Найдено PDF: {len(pdf_files)}")
    for pdf in pdf_files:
        print(f"   • {os.path.basename(pdf)}")
    print()

    # 2. Создаём конвертер и чанкер
    print("🔧 Инициализация...")
    converter = create_converter()
    chunker = create_chunker(embed_model, max_tokens)
    print()

    # 3. Конвертируем и чанким каждый PDF
    all_documents = []
    
    for pdf_path in pdf_files:
        print(f"📖 Обработка: {os.path.basename(pdf_path)}")
        
        # Конвертация (с кешированием)
        doc = convert_pdf_to_docling(pdf_path, converter, cache_dir)
        
        # Chunking
        source_name = os.path.basename(pdf_path)
        documents = chunk_document(doc, chunker, source_name)
        print(f"  ✂️ Создано чанков: {len(documents)}")
        
        all_documents.extend(documents)
        print()

    print(f"📊 Всего чанков: {len(all_documents)}")
    print()

    # 4. Debug dump
    if debug_dump:
        chunks_path = "chunks.jsonl"
        print(f"🔍 DEBUG: Сохранение в {chunks_path}...")
        save_chunks_for_debug(all_documents, chunks_path)
        print()

    # 5. Строим индекс
    print("🔨 Построение индекса...")
    build_index(all_documents, index_dir, embed_model)
    print()

    # 6. Сохраняем конфиг
    save_index_config(
        index_dir=index_dir,
        embed_model=embed_model,
        max_tokens=max_tokens,
        pdf_files=pdf_files,
        chunk_count=len(all_documents),
    )

    print()
    print("=" * 60)
    print("✅ Индекс успешно создан!")
    print("=" * 60)
    return True


def main():
    """CLI точка входа."""
    books_dir = os.getenv("BOOKS_DIR", DEFAULT_BOOKS_DIR)
    index_dir = os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR)
    
    rebuild_full_index(books_dir, index_dir)


if __name__ == "__main__":
    main()
