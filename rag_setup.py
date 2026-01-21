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


DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BOOKS_DIR = "books/"
DEFAULT_INDEX_DIR = "rag_index/"
USE_MARKDOWN_SPLITTER = True


def is_e5_model(model_name: str) -> bool:
    return "e5" in model_name.lower()


def add_passage_prefix(chunks: list, embed_model: str) -> list:
    if not is_e5_model(embed_model):
        return chunks
    print(f"  📝 Добавляем 'passage:' префикс для E5 модели")
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"
    return chunks


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_pdf_files(books_dir: str) -> list[str]:
    pattern = os.path.join(books_dir, "*.pdf")
    return sorted(glob.glob(pattern))


def load_documents_with_docling(pdf_paths: list[str]) -> list[Document]:
    converter = DocumentConverter()
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"  📄 Docling конвертирует: {pdf_path}")
        try:
            result = converter.convert(pdf_path)
            markdown_content = result.document.export_to_markdown()
            doc = Document(
                page_content=markdown_content,
                metadata={"source": pdf_path, "page": 0, "converter": "docling"}
            )
            all_docs.append(doc)
            print(f"    ✅ Сконвертировано: {len(markdown_content)} символов")
        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
    return all_docs


def split_documents(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    if USE_MARKDOWN_SPLITTER:
        print(f"  📝 Используем MarkdownTextSplitter")
        splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def save_chunks_for_debug(chunks: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            record = {"chunk_id": i, "source": chunk.metadata.get("source", ""), "text": chunk.page_content}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_index_config(index_dir, chunk_size, chunk_overlap, embed_model, pdf_files, chunk_count):
    config = {
        "embed_model": embed_model, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
        "built_at": datetime.now().isoformat(), "pdf_count": len(pdf_files),
        "chunk_count": chunk_count, "pdf_parser": "docling"
    }
    with open(os.path.join(index_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def build_index(chunks: list, index_dir: str, embed_model: str) -> None:
    chunks = add_passage_prefix(chunks, embed_model)
    print(f"  🔢 Создание embeddings ({embed_model})...")
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    print("  📦 Построение FAISS индекса...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)
    print(f"  💾 Индекс сохранён: {index_dir}")


def rebuild_full_index(books_dir: str, index_dir: str) -> bool:
    chunk_size = get_env_int("CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = get_env_int("CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)
    embed_model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    debug_dump = os.getenv("DEBUG_DUMP_CHUNKS", "0") == "1"

    print("=" * 50)
    print("🚀 Сборка FAISS индекса (Docling + Markdown)")
    print("=" * 50)
    print(f"  📁 PDF: {books_dir}, Индекс: {index_dir}")
    print(f"  📏 chunk_size: {chunk_size}, overlap: {chunk_overlap}")
    print(f"  🧠 embed_model: {embed_model}")
    print()

    pdf_files = get_pdf_files(books_dir)
    if not pdf_files:
        print(f"❌ Нет PDF в {books_dir}")
        return False

    print(f"📚 Найдено PDF: {len(pdf_files)}")
    print()
    print("📖 Конвертация через Docling...")
    docs = load_documents_with_docling(pdf_files)
    if not docs:
        return False
    print()
    print("✂️ Чанкинг...")
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    print(f"   Создано чанков: {len(chunks)}")
    print()
    if debug_dump:
        save_chunks_for_debug(chunks, "chunks.jsonl")
        print("🔍 Чанки сохранены в chunks.jsonl")
    print("🔨 Построение индекса...")
    build_index(chunks, index_dir, embed_model)
    save_index_config(index_dir, chunk_size, chunk_overlap, embed_model, pdf_files, len(chunks))
    print()
    print("✅ Готово!")
    return True


def main():
    rebuild_full_index(os.getenv("BOOKS_DIR", DEFAULT_BOOKS_DIR), os.getenv("INDEX_DIR", DEFAULT_INDEX_DIR))


if __name__ == "__main__":
    main()
