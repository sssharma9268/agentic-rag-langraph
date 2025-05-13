"""Main FastAPI application module."""

import os
import asyncio
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
import aiofiles

from .document_processor import DocumentProcessor
from .vector_store import VectorStoreManager
from .rag_agent import RAGAgent

# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="API for document ingestion and querying using RAG",
    version="0.1.0"
)

# Initialize components
PERSIST_DIRECTORY = "vector_store"
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

document_processor = DocumentProcessor()
vector_store = VectorStoreManager(persist_directory=PERSIST_DIRECTORY)
rag_agent = RAGAgent(vector_store=vector_store)

class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    run_tree: Optional[dict] = None

async def process_file_async(file_path: str, file_type: str, file_name: str) -> None:
    """Process a file asynchronously.
    
    Args:
        file_path: Path to the temporary file
        file_type: Type of the file
        file_name: Original file name
    """
    try:
        documents = await document_processor.process_file(file_path, file_type)
        await vector_store.add_documents(documents)
    finally:
        # Clean up temporary file
        os.unlink(file_path)

@app.post("/documents", status_code=201)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process documents endpoint asynchronously.
    
    Args:
        background_tasks: FastAPI background tasks handler
        files: List of files to process
        
    Returns:
        Dictionary with processing status
    """
    processed_files = []
    
    for file in files:
        # Get file extension
        file_extension = Path(file.filename).suffix.lower().lstrip(".")
        
        # Validate file type
        if file_extension not in ["pdf", "docx", "txt", "json", "xlsx", "html"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Save file temporarily
        temp_file = NamedTemporaryFile(delete=False, suffix=f".{file_extension}")
        try:
            async with aiofiles.open(temp_file.name, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Add file processing to background tasks
            background_tasks.add_task(
                process_file_async,
                temp_file.name,
                file_extension,
                file.filename
            )
            
            processed_files.append(file.filename)
            
        except Exception as e:
            # Clean up temp file in case of error
            os.unlink(temp_file.name)
            raise HTTPException(
                status_code=500,
                detail=f"Error processing file {file.filename}: {str(e)}"
            )
    
    return {
        "status": "success",
        "message": f"Processing {len(processed_files)} files in background",
        "processed_files": processed_files
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents endpoint asynchronously.
    
    Args:
        request: Query request containing question and optional conversation ID
        
    Returns:
        Answer and run tree from the RAG agent
    """
    try:
        response = await rag_agent.process_question(
            request.question,
            request.conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) 