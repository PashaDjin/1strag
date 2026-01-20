#!/bin/bash
# Quick start script for RAG chatbot

echo "🚀 Starting RAG Chatbot Setup..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+"
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Make sure Ollama is installed (https://ollama.ai)"
echo "2. Run: ollama pull llama3"
echo "3. Add PDF files to the books/ directory"
echo "4. Verify setup: python3 verify_setup.py"
echo "5. Start the app: streamlit run app.py"
