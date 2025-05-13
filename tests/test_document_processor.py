"""Tests for document processor module."""

import pytest
from pathlib import Path
import json
import pandas as pd
from app.document_processor import DocumentProcessor

@pytest.fixture
def document_processor():
    """Create a document processor instance."""
    return DocumentProcessor(chunk_size=100, chunk_overlap=20)

def test_process_text_file(document_processor, tmp_path):
    """Test processing of text files."""
    # Create a test text file
    text_content = "This is a test document.\nIt has multiple lines.\nTesting text processing."
    text_file = tmp_path / "test.txt"
    text_file.write_text(text_content)
    
    # Process the file
    documents = document_processor.process_file(text_file, "txt")
    
    # Assertions
    assert len(documents) > 0
    assert all(doc.page_content for doc in documents)
    assert all(doc.metadata["source"] == str(text_file) for doc in documents)

def test_process_json_file(document_processor, tmp_path):
    """Test processing of JSON files."""
    # Create a test JSON file
    json_content = {
        "title": "Test Document",
        "content": "This is a test JSON document",
        "metadata": {"author": "Test Author"}
    }
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(json_content))
    
    # Process the file
    documents = document_processor.process_file(json_file, "json")
    
    # Assertions
    assert len(documents) > 0
    assert all(doc.page_content for doc in documents)
    assert all(doc.metadata["source"] == str(json_file) for doc in documents)

def test_process_excel_file(document_processor, tmp_path):
    """Test processing of Excel files."""
    # Create a test Excel file
    df = pd.DataFrame({
        "Column1": ["Row1", "Row2", "Row3"],
        "Column2": ["Data1", "Data2", "Data3"]
    })
    excel_file = tmp_path / "test.xlsx"
    df.to_excel(excel_file, index=False)
    
    # Process the file
    documents = document_processor.process_file(excel_file, "xlsx")
    
    # Assertions
    assert len(documents) > 0
    assert all(doc.page_content for doc in documents)
    assert all(doc.metadata["source"] == str(excel_file) for doc in documents)

def test_unsupported_file_type(document_processor, tmp_path):
    """Test handling of unsupported file types."""
    # Create a test file with unsupported extension
    test_file = tmp_path / "test.xyz"
    test_file.write_text("Some content")
    
    # Assert that processing raises ValueError
    with pytest.raises(ValueError, match="Unsupported file type"):
        document_processor.process_file(test_file, "xyz")

def test_chunk_size_and_overlap(document_processor, tmp_path):
    """Test document chunking with specific size and overlap."""
    # Create a long text file
    long_text = " ".join(["word" + str(i) for i in range(100)])
    text_file = tmp_path / "long.txt"
    text_file.write_text(long_text)
    
    # Process the file
    documents = document_processor.process_file(text_file, "txt")
    
    # Assertions
    assert len(documents) > 1  # Should create multiple chunks
    # Check that chunks are roughly the expected size
    for doc in documents:
        assert len(doc.page_content) <= document_processor.chunk_size + 50  # Allow some flexibility 