import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

def load_pdfs(books_dir="books"):
    """Load all PDF files from the books directory."""
    documents = []
    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"Books directory '{books_dir}' not found")
    
    pdf_files = [f for f in os.listdir(books_dir) if f.endswith('.pdf')]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{books_dir}'")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(books_dir, pdf_file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())
    
    return documents

def create_vector_store(documents):
    """Split documents and create FAISS vector store."""
    # Split with 1000 char chunks, 200 char overlap as specified
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model="llama3")
    
    # Create and save FAISS vector store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    
    return vectorstore

def load_vector_store():
    """Load existing FAISS vector store."""
    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def create_qa_chain(vectorstore):
    """Create RetrievalQA chain with Ollama."""
    llm = Ollama(model="llama3")
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

def setup_rag():
    """Main setup function - load or create vector store."""
    if os.path.exists("faiss_index"):
        print("Loading existing vector store...")
        vectorstore = load_vector_store()
    else:
        print("Creating new vector store...")
        documents = load_pdfs()
        print(f"Loaded {len(documents)} pages from PDFs")
        vectorstore = create_vector_store(documents)
        print("Vector store created and saved")
    
    return create_qa_chain(vectorstore)
