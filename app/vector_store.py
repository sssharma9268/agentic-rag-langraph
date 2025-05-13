"""Vector store management module."""

from typing import List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[Embeddings] = None
    ):
        """Initialize the vector store manager.
        
        Args:
            persist_directory: Directory to persist the vector store
            embedding_model: Embedding model to use
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize the vector store."""
        if self.persist_directory:
            return Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model
            )
        return Chroma(embedding_function=self.embedding_model)
    
    async def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store asynchronously.
        
        Args:
            documents: List of documents to add
        """
        # Since Chroma's add_documents is not async, we'll keep the operation
        # but wrap it in an async function for consistency
        self.vector_store.add_documents(documents)
        if self.persist_directory:
            self.vector_store.persist()
    
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform similarity search asynchronously.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional filter criteria
            
        Returns:
            List of relevant documents
        """
        # Since Chroma's similarity_search is not async, we'll keep the operation
        # but wrap it in an async function for consistency
        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
    
    async def hybrid_search(
        self,
        query: str,
        k: int = 4,
        alpha: float = 0.5,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """Perform hybrid search asynchronously.
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for hybrid search (0 = BM25 only, 1 = semantic only)
            filter: Optional filter criteria
            
        Returns:
            List of relevant documents
        """
        # Since Chroma's hybrid_search is not async, we'll keep the operation
        # but wrap it in an async function for consistency
        return self.vector_store.hybrid_search(
            query=query,
            k=k,
            alpha=alpha,
            filter=filter
        ) 