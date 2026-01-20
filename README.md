# 1strag - Simple Local RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot using LangChain, Ollama, Streamlit, and FAISS.

## Features
- 📚 Answer questions from PDF books with sources
- 🔍 RecursiveCharacterTextSplitter with 1000/200 chunking
- 💾 FAISS vector store for efficient retrieval
- 🦙 Ollama (llama3) for local LLM inference
- 🎨 Clean Streamlit UI on localhost:8501
- ✅ Codespaces ready

## Setup

### Prerequisites
1. **Install Ollama**: Download from [ollama.ai](https://ollama.ai)
2. **Pull llama3 model**:
   ```bash
   ollama pull llama3
   ```

### Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify setup
python3 verify_setup.py

# Add your PDF books to the books/ directory
cp your_book.pdf books/
```

### Usage
```bash
# Start the chatbot
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Codespaces Deployment

This project is configured for GitHub Codespaces:

1. **Open in Codespaces**: Click "Code" → "Codespaces" → "Create codespace"
2. **Install Ollama**: Follow instructions in `.devcontainer/OLLAMA_SETUP.md`
3. **Add PDFs**: Upload your PDF books to the `books/` directory
4. **Run**: `streamlit run app.py`

The Streamlit port (8501) will be automatically forwarded.

## How It Works

**Flow**: PDF → RecursiveSplitter(1000/200) → FAISS → RetrievalQA → Ollama(llama3) → Streamlit UI

1. **PDF Loading**: PDFs from `books/` directory are loaded and split into pages
2. **Chunking**: Documents are split into 1000-char chunks with 200-char overlap
3. **Embedding**: Chunks are embedded using Ollama embeddings
4. **Vector Store**: FAISS index is created and saved locally
5. **Retrieval**: Top 3 relevant chunks are retrieved for each query
6. **Generation**: Ollama llama3 generates answers based on retrieved context
7. **UI**: Streamlit displays chat interface with sources

## File Structure
```
1strag/
├── app.py                          # Streamlit UI
├── rag_setup.py                    # RAG configuration
├── requirements.txt                # Dependencies
├── books/                          # PDF storage
├── AntiOverengineering_RULES.md    # Coding principles
└── README.md                       # This file
```

## Anti-Overengineering
This project follows strict minimalism principles (see `AntiOverengineering_RULES.md`):
- One file = One task
- No unnecessary abstractions
- Direct, simple code
- Minimal dependencies

## Troubleshooting

**No PDFs found**: Add PDF files to the `books/` directory

**Ollama connection error**: Make sure Ollama is running (`ollama serve`)

**Model not found**: Pull the llama3 model (`ollama pull llama3`)

## License
MIT