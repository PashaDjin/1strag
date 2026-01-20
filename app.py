import streamlit as st
from rag_setup import setup_rag

# Page config
st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

# Title
st.title("📚 Local RAG Chatbot")
st.caption("Ask questions about your PDF books")

# Initialize QA chain
@st.cache_resource
def load_qa_chain():
    """Load QA chain once and cache it."""
    try:
        return setup_rag()
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        return None

qa_chain = load_qa_chain()

# Chat interface
if qa_chain:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📄 Sources"):
                    st.write(message["sources"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your books..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain({"query": prompt})
                answer = response["result"]
                source_docs = response["source_documents"]
                
                # Format sources
                sources = []
                for i, doc in enumerate(source_docs, 1):
                    source_info = f"**Source {i}:**\n"
                    source_info += f"- File: {doc.metadata.get('source', 'Unknown')}\n"
                    source_info += f"- Page: {doc.metadata.get('page', 'Unknown')}\n"
                    source_info += f"- Content: {doc.page_content[:200]}..."
                    sources.append(source_info)
                
                sources_text = "\n\n".join(sources)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources in expander
                with st.expander("📄 Sources"):
                    st.write(sources_text)
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources_text
                })
else:
    st.error("Failed to initialize RAG system. Please check your setup.")
    st.info("Make sure you have:\n1. PDF files in the 'books/' directory\n2. Ollama running with llama3 model installed")
