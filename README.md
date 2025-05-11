# Document Intelligence App

A Streamlit application for document processing, OCR, and question answering using LangChain and RAG.

## Features

- Upload and process PDF documents
- Extract text using OCR (Optical Character Recognition)
- Store documents in a vector database (Chroma)
- Search documents using semantic search
- Ask questions about your documents using RAG (Retrieval Augmented Generation)

## Requirements

The application requires several dependencies, which are listed in `requirements.txt`. The main dependencies are:

- streamlit
- langchain, langchain-community, langchain-text-splitters, etc.
- langgraph
- langchain-mistralai, langchain-huggingface
- langchain-chroma
- pypdf, PyPDF2
- pytesseract
- pdf2image
- chromadb

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:

```bash
streamlit run app.py
```

2. Configure API keys in the sidebar
3. Upload and process documents
4. Search or ask questions about your documents

## Project Structure

- `app.py`: Main Streamlit application
- `ocr_utils.py`: OCR-related functions
- `vector_store.py`: Vector store and document management functions
- `rag_pipeline.py`: RAG pipeline implementation
- `requirements.txt`: List of dependencies

## Notes

- The app requires a Mistral API key for the RAG pipeline to work
- OCR functionality requires Tesseract and Poppler to be installed on the system
- All data is stored locally in a temporary directory during the session