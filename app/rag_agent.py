"""RAG agent module using Langraph for orchestration."""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langgraph.graph import Graph
from langsmith.run_trees import RunTree

from .vector_store import VectorStoreManager

class RAGAgent:
    """RAG agent using Langraph for orchestration."""
    
    def __init__(
        self,
        vector_store: VectorStoreManager,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Initialize the RAG agent.
        
        Args:
            vector_store: Vector store manager instance
            model_name: Name of the LLM to use
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question. "
                      "If you cannot find the answer in the context, say so."),
            ("context", "{context}"),
            ("human", "{question}")
        ])
        
        self.graph = self._create_graph()
    
    def _create_graph(self) -> Graph:
        """Create the Langraph processing graph."""
        retrieval_chain = (
            {"context": self._retrieve_documents, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        return Graph.from_chain(
            retrieval_chain,
            name="rag_agent",
            description="RAG agent for document question answering"
        )
    
    def _retrieve_documents(self, question: str) -> str:
        """Retrieve relevant documents for the question.
        
        Args:
            question: User's question
            
        Returns:
            Formatted context string
        """
        docs = self.vector_store.hybrid_search(question)
        return "\n\n".join(doc.page_content for doc in docs)
    
    async def process_question(
        self,
        question: str,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a user question and return the response.
        
        Args:
            question: User's question
            conversation_id: Optional conversation ID for tracking
            
        Returns:
            Dictionary containing response and metadata
        """
        config = {"configurable": {"conversation_id": conversation_id}} if conversation_id else {}
        
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        
        return {
            "answer": result.messages[-1].content,
            "run_tree": result.run_tree.dict() if result.run_tree else None
        } 