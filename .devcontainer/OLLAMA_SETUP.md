# Installation Script for Ollama in Codespaces

This script helps you install and run Ollama in GitHub Codespaces.

## Quick Install

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama in background
ollama serve &

# Wait a few seconds for Ollama to start
sleep 5

# Pull llama3 model
ollama pull llama3

# Verify installation
python3 verify_setup.py
```

## Manual Steps

If the automatic installation doesn't work, follow these steps:

1. **Download Ollama**:
   ```bash
   curl -L https://ollama.ai/download/ollama-linux-amd64 -o /usr/local/bin/ollama
   chmod +x /usr/local/bin/ollama
   ```

2. **Start Ollama**:
   ```bash
   ollama serve &
   ```

3. **Pull llama3**:
   ```bash
   ollama pull llama3
   ```

4. **Verify**:
   ```bash
   python3 verify_setup.py
   ```

## Running the App

Once Ollama is set up:

```bash
streamlit run app.py
```

The app will be available at http://localhost:8501
