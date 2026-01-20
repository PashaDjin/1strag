# Security Update - January 2026

## Vulnerabilities Fixed

Updated `langchain-community` from version 0.0.16 to 0.3.27 to address the following security vulnerabilities:

### 1. XML External Entity (XXE) Attacks
- **Severity**: High
- **Affected versions**: < 0.3.27
- **Patched version**: 0.3.27
- **Description**: Langchain Community was vulnerable to XML External Entity (XXE) attacks that could allow attackers to read arbitrary files or cause denial of service.

### 2. SSRF Vulnerability in RequestsToolkit
- **Severity**: Medium
- **Affected versions**: < 0.0.28
- **Patched version**: 0.0.28
- **Description**: Server-Side Request Forgery vulnerability in the RequestsToolkit component that could allow unauthorized network requests.

### 3. Pickle Deserialization of Untrusted Data
- **Severity**: High
- **Affected versions**: < 0.2.4
- **Patched version**: 0.2.4
- **Description**: Unsafe pickle deserialization that could lead to arbitrary code execution.

## Updated Dependencies

- `langchain`: 0.1.4 → 0.3.27
- `langchain-community`: 0.0.16 → 0.3.27
- `langchain-core`: 0.1.23 → 0.3.83 (auto-updated)
- `langchain-text-splitters`: Added 0.3.11 (new dependency)

## Verification

All functionality has been tested and verified to work with the updated versions:

- ✓ PDF loading and processing
- ✓ Text splitting (RecursiveCharacterTextSplitter)
- ✓ FAISS vector store operations
- ✓ Ollama LLM integration
- ✓ RetrievalQA chain
- ✓ Streamlit UI

## Notes

The update maintains backward compatibility with the existing codebase. No code changes were required, only dependency version updates.
