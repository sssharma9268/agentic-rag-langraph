# Agentic RAG Application

An advanced RAG (Retrieval-Augmented Generation) application built using Langchain, Langraph, and Model Context Protocol. This application provides an efficient way to process, store, and query various document types using state-of-the-art language models and vector databases.

## Features

- Document ingestion support for multiple formats:
  - PDF
  - Word Documents
  - Text Files
  - JSON
  - Excel
  - HTML Pages
- OCR capabilities for processing scanned documents
- Efficient document chunking and embedding
- Vector storage using Chroma/FAISS
- Advanced retrieval strategies:
  - Similarity Search
  - Hybrid Search
- FastAPI endpoints for document ingestion and querying
- Comprehensive test coverage

## Installation

1. Clone the repository
2. Install Poetry (if not already installed)
3. Install dependencies:
```bash
poetry install
```

## Usage

1. Start the FastAPI server:
```bash
poetry run uvicorn app.main:app --reload
```

2. API Endpoints:
- POST `/documents`: Upload documents for processing
- POST `/query`: Query the processed documents

## Development

Run tests:
```bash
poetry run pytest
```

Generate test coverage report:
```bash
poetry run pytest --cov=app tests/
```

## Requirements

- Python 3.10+
- Poetry
- Tesseract OCR (for OCR capabilities) 