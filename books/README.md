# Books Directory

Place your PDF files in this directory to create your RAG knowledge base.

## Supported Formats
- PDF files only (`.pdf` extension)

## Examples
```bash
# Copy a PDF to this directory
cp ~/Documents/mybook.pdf .

# Download a PDF
wget https://example.com/book.pdf
```

## Notes
- PDF files in this directory are excluded from git (see `.gitignore`)
- The system will automatically process all PDFs when you run the app for the first time
- After processing, a `faiss_index/` directory will be created with the vector embeddings

## Getting Started
1. Add at least one PDF file to this directory
2. Run `python3 verify_setup.py` from the project root to verify setup
3. Run `streamlit run app.py` to start the chatbot
