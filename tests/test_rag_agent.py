"""Tests for RAG agent and FastAPI endpoints."""

import pytest
import os
from pathlib import Path
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.main import app
from app.rag_agent import RAGAgent
from app.vector_store import VectorStoreManager

@pytest.fixture
def test_client():
    """Create a test client for FastAPI app."""
    return TestClient(app)

@pytest.fixture
def vector_store(tmp_path):
    """Create a vector store instance for testing."""
    return VectorStoreManager(persist_directory=str(tmp_path / "test_vector_store"))

@pytest.fixture
def rag_agent(vector_store):
    """Create a RAG agent instance for testing."""
    return RAGAgent(vector_store=vector_store)

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The capital of France is Paris. It is known for the Eiffel Tower.",
            metadata={"source": "geography.txt"}
        ),
        Document(
            page_content="Python is a popular programming language created by Guido van Rossum.",
            metadata={"source": "programming.txt"}
        )
    ]

@pytest.mark.asyncio
async def test_rag_agent_question_processing(rag_agent, vector_store, sample_documents):
    """Test RAG agent's question processing."""
    # Add documents to vector store
    vector_store.add_documents(sample_documents)
    
    # Test question about content in documents
    response = await rag_agent.process_question("What is the capital of France?")
    assert response["answer"] is not None
    assert "Paris" in response["answer"]
    
    # Test question about content not in documents
    response = await rag_agent.process_question("What is the population of Tokyo?")
    assert response["answer"] is not None
    assert any(phrase in response["answer"].lower() 
              for phrase in ["don't know", "cannot find", "no information"])

def test_document_upload_endpoint(test_client, tmp_path):
    """Test document upload endpoint."""
    # Create a test text file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document for upload.")
    
    # Upload file
    with open(test_file, "rb") as f:
        response = test_client.post(
            "/documents",
            files={"files": ("test.txt", f, "text/plain")}
        )
    
    assert response.status_code == 201
    assert response.json()["status"] == "success"
    assert len(response.json()["processed_files"]) == 1

def test_query_endpoint(test_client, vector_store, sample_documents):
    """Test query endpoint."""
    # Add documents to vector store
    vector_store.add_documents(sample_documents)
    
    # Test valid query
    response = test_client.post(
        "/query",
        json={"question": "What is Python?"}
    )
    
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] is not None
    
    # Test query with conversation ID
    response = test_client.post(
        "/query",
        json={
            "question": "What is Python?",
            "conversation_id": "test-conv-1"
        }
    )
    
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] is not None

def test_invalid_file_upload(test_client, tmp_path):
    """Test upload of invalid file type."""
    # Create an invalid file
    invalid_file = tmp_path / "test.xyz"
    invalid_file.write_text("Invalid file content")
    
    # Try to upload invalid file
    with open(invalid_file, "rb") as f:
        response = test_client.post(
            "/documents",
            files={"files": ("test.xyz", f, "application/octet-stream")}
        )
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_error_handling_query_endpoint(test_client):
    """Test error handling in query endpoint."""
    # Test empty question
    response = test_client.post(
        "/query",
        json={"question": ""}
    )
    
    assert response.status_code != 200
    
    # Test missing question field
    response = test_client.post(
        "/query",
        json={}
    )
    
    assert response.status_code != 200 