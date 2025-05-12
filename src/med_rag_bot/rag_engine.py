"""
RAG engine module for medical RAG bot.
Integrates embedding generation, vector store querying, and LLM response generation.
"""

from typing import Dict, Any#, List, Optional
#import re
#from tqdm import tqdm

from med_rag_bot.embeddings import EmbeddingGenerator
from med_rag_bot.vector_store import VectorStore
from med_rag_bot.llm import GeminiLLM
from med_rag_bot.pdf_processor import PDFProcessor

class RAGEngine:
    """RAG (Retrieval-Augmented Generation) Engine that integrates all components"""
    
    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        vector_store: VectorStore,
        llm: GeminiLLM,
        top_k: int = 5,
        similarity_threshold: float = 0.75
    ):
        """
        Initialize RAG Engine.
        
        Args:
            embedding_generator: Component for generating embeddings
            vector_store: Component for storing and retrieving vector embeddings
            llm: Component for generating responses
            top_k: Number of similar documents to retrieve
            similarity_threshold: Minimum similarity score to consider relevant
        """
        self.embedding_generator = embedding_generator
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        print("Initialized RAG Engine with components:")
        print(f"- Embedding Generator: {self.embedding_generator.__class__.__name__}")
        print(f"- Vector Store: {self.vector_store.index_name}")
        print(f"- LLM: {self.llm.model_name}")
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> bool:
        """
        Process a PDF file and store embeddings in the vector store.
        
        Args:
            pdf_path: Path to the PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            bool: Success status
        """
        try:
            # Process PDF into chunks
            print(f"Processing PDF: {pdf_path}")
            processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = processor.process_pdf(pdf_path)
            
            # Generate embeddings for all chunks
            print("Generating embeddings for chunks...")
            texts = [chunk["text"] for chunk in chunks]
            
            # Process in smaller batches to avoid memory issues with large PDFs
            batch_size = 32
            all_embeddings = []
            
            total_batches = (len(texts) + batch_size - 1) // batch_size
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{total_batches}")
                batch_embeddings = self.embedding_generator.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)
            
            # Combine chunks with their embeddings
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = all_embeddings[i]
            
            # Store in vector database
            print("Storing embeddings in vector database...")
            success = self.vector_store.upsert_items(chunks)
            
            if success:
                print(f"Successfully ingested PDF with {len(chunks)} chunks")
            else:
                print("Failed to ingest PDF")
            
            return success
        
        except Exception as e:
            print(f"Error ingesting PDF: {e}")
            return False
    
    def query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the RAG pipeline.
        
        Args:
            query: User's query text
            
        Returns:
            Dict with query results and metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_generator.get_embedding(query)
            
            # Retrieve similar documents
            similar_docs = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.top_k
            )
            
            # Check if any retrieved documents are relevant based on similarity threshold
            has_relevant_docs = any(doc["score"] >= self.similarity_threshold for doc in similar_docs)
            
            if not has_relevant_docs or not similar_docs:
                # If no relevant documents found, generate a standard response
                response = self.llm.generate_response(query)
                return {
                    "query": query,
                    "response": response,
                    "has_context": False,
                    "sources": [],
                    "context": "",
                    "is_out_of_scope": True
                }
            
            # Combine the retrieved documents into context
            context = "\n\n".join([f"Excerpt {i+1}:\n{doc['text']}" for i, doc in enumerate(similar_docs)])
            
            # Generate response using the context
            response = self.llm.generate_response(query, context)
            
            return {
                "query": query,
                "response": response,
                "has_context": True,
                "sources": similar_docs,
                "context": context,
                "is_out_of_scope": False
            }
        
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                "query": query,
                "response": "I'm sorry, but I encountered an error processing your query. Please try again.",
                "has_context": False,
                "sources": [],
                "context": "",
                "is_out_of_scope": True,
                "error": str(e)
            }
