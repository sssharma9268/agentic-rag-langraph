"""Tests for vector store manager module."""

import pytest
from pathlib import Path
from langchain_core.documents import Document
from app.vector_store import VectorStoreManager

@pytest.fixture
def vector_store(tmp_path):
    """Create a vector store instance."""
    return VectorStoreManager(persist_directory=str(tmp_path / "test_vector_store"))

@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The quick brown fox jumps over the lazy dog",
            metadata={"source": "test1.txt"}
        ),
        Document(
            page_content="Pack my box with five dozen liquor jugs",
            metadata={"source": "test2.txt"}
        ),
        Document(
            page_content="How vexingly quick daft zebras jump",
            metadata={"source": "test3.txt"}
        )
    ]

def test_add_documents(vector_store, sample_documents):
    """Test adding documents to vector store."""
    # Add documents
    vector_store.add_documents(sample_documents)
    
    # Search to verify documents were added
    results = vector_store.similarity_search("fox", k=1)
    assert len(results) == 1
    assert "fox" in results[0].page_content.lower()

def test_similarity_search(vector_store, sample_documents):
    """Test similarity search functionality."""
    # Add documents
    vector_store.add_documents(sample_documents)
    
    # Test search with different k values
    results_k1 = vector_store.similarity_search("quick", k=1)
    results_k2 = vector_store.similarity_search("quick", k=2)
    
    assert len(results_k1) == 1
    assert len(results_k2) == 2
    assert all("quick" in doc.page_content.lower() for doc in results_k2)

def test_hybrid_search(vector_store, sample_documents):
    """Test hybrid search functionality."""
    # Add documents
    vector_store.add_documents(sample_documents)
    
    # Test hybrid search with different alpha values
    results_semantic = vector_store.hybrid_search("animal jumping", k=1, alpha=1.0)
    results_keyword = vector_store.hybrid_search("animal jumping", k=1, alpha=0.0)
    results_hybrid = vector_store.hybrid_search("animal jumping", k=1, alpha=0.5)
    
    assert len(results_semantic) == 1
    assert len(results_keyword) == 1
    assert len(results_hybrid) == 1

def test_persistence(tmp_path):
    """Test vector store persistence."""
    persist_dir = str(tmp_path / "persist_test")
    
    # Create first instance and add documents
    store1 = VectorStoreManager(persist_directory=persist_dir)
    docs = [Document(page_content="Test document", metadata={"source": "test.txt"})]
    store1.add_documents(docs)
    
    # Create second instance pointing to same directory
    store2 = VectorStoreManager(persist_directory=persist_dir)
    
    # Verify document is retrievable from second instance
    results = store2.similarity_search("test", k=1)
    assert len(results) == 1
    assert "test" in results[0].page_content.lower()

def test_filter_search(vector_store, sample_documents):
    """Test search with metadata filtering."""
    # Add documents
    vector_store.add_documents(sample_documents)
    
    # Search with filter
    filter_dict = {"source": "test1.txt"}
    results = vector_store.similarity_search("quick", k=2, filter=filter_dict)
    
    assert len(results) > 0
    assert all(doc.metadata["source"] == "test1.txt" for doc in results) 