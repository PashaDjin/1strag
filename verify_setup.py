#!/usr/bin/env python3
"""
Verify RAG setup before running the application.
"""
import sys
import os

def check_ollama():
    """Check if Ollama is installed and running."""
    import subprocess
    try:
        result = subprocess.run(['ollama', '--version'], 
                                capture_output=True, 
                                text=True, 
                                timeout=5)
        if result.returncode == 0:
            print("✓ Ollama is installed")
            return True
        else:
            print("✗ Ollama is not installed")
            return False
    except FileNotFoundError:
        print("✗ Ollama is not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {e}")
        return False

def check_llama3():
    """Check if llama3 model is available."""
    import subprocess
    try:
        result = subprocess.run(['ollama', 'list'], 
                                capture_output=True, 
                                text=True, 
                                timeout=5)
        if 'llama3' in result.stdout:
            print("✓ llama3 model is available")
            return True
        else:
            print("✗ llama3 model is not available")
            print("  Run: ollama pull llama3")
            return False
    except Exception as e:
        print(f"✗ Error checking llama3: {e}")
        return False

def check_pdfs():
    """Check if PDFs exist in books directory."""
    books_dir = "books"
    if not os.path.exists(books_dir):
        print("✗ books/ directory not found")
        return False
    
    pdfs = [f for f in os.listdir(books_dir) if f.endswith('.pdf')]
    if pdfs:
        print(f"✓ Found {len(pdfs)} PDF file(s) in books/")
        for pdf in pdfs:
            print(f"  - {pdf}")
        return True
    else:
        print("✗ No PDF files found in books/")
        print("  Add PDF files to the books/ directory")
        return False

def check_dependencies():
    """Check if required Python packages are installed."""
    required = ['streamlit', 'langchain', 'langchain_community', 'faiss', 'pypdf', 'ollama']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            missing.append(package)
    
    if missing:
        print(f"\n  Run: pip install -r requirements.txt")
        return False
    return True

def main():
    print("=== RAG Chatbot Setup Verification ===\n")
    
    print("1. Checking Python dependencies...")
    deps_ok = check_dependencies()
    print()
    
    print("2. Checking Ollama installation...")
    ollama_ok = check_ollama()
    print()
    
    if ollama_ok:
        print("3. Checking llama3 model...")
        llama3_ok = check_llama3()
        print()
    else:
        llama3_ok = False
        print("3. Skipping llama3 check (Ollama not installed)\n")
    
    print("4. Checking PDF files...")
    pdfs_ok = check_pdfs()
    print()
    
    print("=== Summary ===")
    if deps_ok and ollama_ok and llama3_ok and pdfs_ok:
        print("✓ All checks passed! Ready to run: streamlit run app.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
