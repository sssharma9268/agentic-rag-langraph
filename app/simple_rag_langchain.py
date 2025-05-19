import os
from getpass import getpass
from typing import List, Tuple, Literal
from langchain_community.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from app.settings import settings

class DocumentProcessor:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def load_from_pdf(self, file_path: str) -> List[Document]:
        """Load documents from local PDF file"""
        try:
            loader = UnstructuredFileLoader(file_path)
            return loader.load()
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return []
    
    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from URL"""
        try:
            loader = UnstructuredURLLoader(urls=[url])
            return loader.load()
        except Exception as e:
            print(f"Error loading URL: {e}")
            return []
    
    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            raise ValueError("No documents to process")
        return self.text_splitter.split_documents(documents)

class VectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self, use_openai=False):
        if use_openai and settings.OPENAI_API_KEY:
            self.embeddings = OpenAIEmbeddings(openai_api_key=settings.OPENAI_API_KEY)
        else:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2"
            )
        self.vector_store = None
    
    def create_index(self, documents: List[Document]):
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents to index")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
    
    def similarity_search(self, query: str, k=5) -> List[Tuple[Document, float]]:
        """Retrieve documents with similarity scores"""
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
            
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
        for doc, score in docs_with_scores:
            doc.metadata["score"] = score
        return docs_with_scores

class RAGPipeline:
    """Orchestrates the complete RAG workflow"""
    
    def __init__(self, llm_provider: Literal["openai", "huggingface"] = "huggingface"):
        """Initialize RAG Pipeline
        
        Args:
            llm_provider: The LLM provider to use ("openai" or "huggingface")
        """
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore(use_openai=llm_provider == "openai")
        
        # Initialize LLM based on provider
        if llm_provider == "openai" and settings.OPENAI_API_KEY:
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                openai_api_key=settings.OPENAI_API_KEY
            )
        elif llm_provider == "huggingface" and settings.HUGGINGFACE_API_KEY:
            self.llm = HuggingFaceEndpoint(
                repo_id="google/flan-t5-xxl",  # You can change this to other models
                huggingfacehub_api_token=settings.HUGGINGFACE_API_KEY,
                temperature=0.7,
                model_kwargs={
                    "max_length": 512
                }
            )
        else:
            raise ValueError(f"Invalid or unconfigured LLM provider: {llm_provider}")
        
        # Create prompt template for response generation
        self.prompt_template = PromptTemplate(
            template="""Use the following context to answer the question. 
            If you cannot find the answer in the context, say so.
            
            Context: {context}
            
            Question: {question}
            
            Answer: Let me help you with that.""",
            input_variables=["context", "question"]
        )
        
        # Create chain using RunnablePassthrough
        self.chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
        )
    
    def ingest_documents(self, source: str, is_url=True):
        """Ingest documents from source"""
        if is_url:
            docs = self.processor.load_from_url(source)
        else:
            docs = self.processor.load_from_pdf(source)
        
        if not docs:
            raise ValueError(f"No documents loaded from source: {source}")
            
        processed_docs = self.processor.process_documents(docs)
        if not processed_docs:
            raise ValueError("Document processing resulted in no chunks")
            
        self.vector_store.create_index(processed_docs)
    
    def retrieve(self, query: str, top_n=5) -> List[Tuple[Document, float]]:
        """Retrieve relevant documents with scores"""
        return self.vector_store.similarity_search(query, k=top_n)
    
    async def generate_response(self, query: str, context: List[Document]) -> str:
        """Generate final response using LLM"""
        if not context:
            raise ValueError("No context provided for response generation")
            
        context_text = "\n\n".join([doc.page_content for doc in context])
        
        # Use the new chain for response generation
        response = await self.chain.ainvoke({
            "context": context_text,
            "question": query
        })
        
        return response.content

# Example usage
if __name__ == "__main__":
    try:
        # Initialize pipeline with HuggingFace (default) or OpenAI
        # rag = RAGPipeline(llm_provider="openai")  # For OpenAI
        rag = RAGPipeline(llm_provider="huggingface")  # For HuggingFace
        
        # Ingest documents (using a more reliable test URL)
        test_url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/README.md"
        rag.ingest_documents(test_url, is_url=True)
        
        # Query the system
        query = "What is LangChain and what are its main features?"
        results = rag.retrieve(query, top_n=3)
        
        # Display results with scores
        print("Top matches:")
        for doc, score in results:
            print(f"Score: {score:.2f}\nContent: {doc.page_content[:200]}...\n")
        
        # Generate final response
        import asyncio
        response = asyncio.run(rag.generate_response(query, [doc for doc, _ in results]))
        print(f"\nFinal response: {response}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
