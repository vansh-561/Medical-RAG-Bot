"""
RAG engine module for medical RAG bot.
Integrates embedding generation, vector store querying, and LLM response generation.
"""

import os
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
    
    def ingest_pdf(self, pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50, timeout: int = 300) -> bool:
        try:
            # Add memory management for large PDFs
            import gc
            import traceback
            import time

            # Release memory before starting
            gc.collect()

            start_time = time.time()
            print(f"Processing PDF: {pdf_path}")
            
            # Calculate appropriate timeout based on file size
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            if timeout is None:
                # 1 minute base + 1 minute per 5MB, capped at 10 minutes
                timeout = min(60 + int(file_size_mb / 5) * 60, 600)
                
            print(f"PDF size: {file_size_mb:.2f}MB - Processing timeout set to {timeout} seconds")
            
            # Process PDF into chunks
            processor = PDFProcessor(
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap,
                timeout=timeout
            )
            
            try:
                chunks = processor.process_pdf(pdf_path)
            except TimeoutError as e:
                print(f"PDF processing timed out: {e}")
                return False
            except MemoryError:
                # If memory error in processor, try with smaller chunks
                print("Memory error in processor. Retrying with smaller chunks...")
                processor = PDFProcessor(
                    chunk_size=250, 
                    chunk_overlap=25,
                    timeout=timeout
                )
                chunks = processor.process_pdf(pdf_path)

            # Safety check
            if not chunks:
                print("No chunks extracted from PDF")
                return False
            
            print(f"Successfully extracted {len(chunks)} chunks from PDF in {time.time() - start_time:.2f} seconds")

            # If processing is taking too long, adjust batch size
            elapsed_so_far = time.time() - start_time
            time_remaining = max(timeout - elapsed_so_far, 60)  # At least 1 minute for embedding
            
            # Generate embeddings for all chunks
            print("Generating embeddings for chunks...")
            texts = [chunk["text"] for chunk in chunks]

            # Process in smaller batches to avoid memory issues
            total_chunks = len(texts)
            # Dynamically adjust batch size based on remaining time and number of chunks
            batch_size = min(8, max(1, int(total_chunks * 60 / time_remaining)))
            print(f"Using batch size of {batch_size} for embeddings generation")
            
            all_embeddings = []

            total_batches = (len(texts) + batch_size - 1) // batch_size
            batch_start_time = time.time()
            
            for i in range(0, len(texts), batch_size):
                current_time = time.time()
                # Check for timeout
                if current_time - start_time > timeout:
                    print(f"Timeout exceeded during embedding generation. Processed {i}/{len(texts)} chunks.")
                    break
                    
                batch_texts = texts[i:i+batch_size]
                print(f"Processing embedding batch {i//batch_size + 1}/{total_batches}")
                batch_embeddings = self.embedding_generator.get_embeddings(batch_texts)
                all_embeddings.extend(batch_embeddings)

                # Report progress and adjust batch size dynamically if needed
                batch_elapsed = time.time() - batch_start_time
                print(f"Batch {i//batch_size + 1} completed in {batch_elapsed:.2f} seconds")
                
                # Adjust batch size if processing is too slow
                if i > 0 and batch_elapsed > 10:  # If a batch takes more than 10 seconds
                    new_batch_size = max(1, batch_size // 2)
                    if new_batch_size != batch_size:
                        print(f"Adjusting batch size from {batch_size} to {new_batch_size}")
                        batch_size = new_batch_size
                        total_batches = (len(texts[i:]) + batch_size - 1) // batch_size + i//batch_size + 1
                
                batch_start_time = time.time()
                # Force garbage collection after each batch
                gc.collect()

            # Check if we have enough embeddings
            if len(all_embeddings) < len(chunks):
                print(f"Warning: Not all chunks were embedded ({len(all_embeddings)}/{len(chunks)})")
                # Only use chunks that have embeddings
                chunks = chunks[:len(all_embeddings)]
                if not chunks:
                    print("No chunks were successfully embedded")
                    return False

            # Combine chunks with their embeddings
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = all_embeddings[i]

            print(f"Successfully generated embeddings in {time.time() - start_time:.2f} seconds")

            # Store in vector database in smaller batches
            print("Storing embeddings in vector database...")
            db_batch_size = min(20, len(chunks))  # Smaller batches for database operations
            success = True

            db_start_time = time.time()
            for i in range(0, len(chunks), db_batch_size):
                current_time = time.time()
                # Check for timeout
                if current_time - start_time > timeout:
                    print(f"Timeout exceeded during database storage. Processed {i}/{len(chunks)} chunks.")
                    break
                    
                batch_chunks = chunks[i:i+db_batch_size]
                print(f"Storing batch {i//db_batch_size + 1}/{(len(chunks) + db_batch_size - 1) // db_batch_size}")
                batch_success = self.vector_store.upsert_items(batch_chunks)
                
                if not batch_success:
                    success = False
                    print(f"Failed to store batch {i//db_batch_size + 1}")

                # Release memory after each database batch
                del batch_chunks
                gc.collect()
                
                # Report progress
                if i % (db_batch_size * 5) == 0 and i > 0:
                    elapsed = time.time() - db_start_time
                    print(f"Database storage progress: {i}/{len(chunks)} chunks in {elapsed:.2f} seconds")
                    db_start_time = time.time()

            # Clean up large objects
            del chunks
            del all_embeddings
            del texts
            gc.collect()

            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            
            if success:
                print("Successfully ingested PDF")
            else:
                print("Failed to completely ingest PDF")

            return success

        except TimeoutError as e:
            print(f"Timeout error ingesting PDF: {e}")
            gc.collect()
            return False
        except MemoryError as e:
            print(f"Memory error ingesting PDF: {e}")
            # Try to recover memory
            import traceback
            traceback.print_exc()
            gc.collect()
            return False
        except Exception as e:
            print(f"Error ingesting PDF: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
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
