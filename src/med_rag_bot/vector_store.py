"""
Vector store module for medical RAG bot.
Handles interactions with Pinecone vector database.
"""

import os
from typing import List, Dict, Any#, Union
#from tqdm import tqdm
#import pinecone
from pinecone import Pinecone, ServerlessSpec

class VectorStore:
    """Interface to Pinecone vector database for storing and retrieving embeddings"""
    
    def __init__(
        self, 
        api_key: str = None, 
        index_name: str = "medical-book",
        namespace: str = "med-textbook",
        dimension: int = 384  # Default for all-MiniLM-L6-v2
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            namespace: Namespace within the index
            dimension: Dimension of embedding vectors
        """
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        self.index_name = index_name
        self.namespace = namespace
        self.dimension = dimension
        
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=self.api_key)
        
        # Check if index exists, create if not
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
        print(f"Connected to Pinecone index: {self.index_name}")
    
    def upsert_items(self, items: List[Dict[str, Any]]) -> bool:
        """
        Insert or update items in the vector database.
        
        Args:
            items: List of items with 'id', 'text', and other metadata
            
        Returns:
            bool: Success status
        """
        try:
            #vectors = []
            batch_size = 100  # Pinecone batch size limit
            
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i+batch_size]
                batch_vectors = [
                    {
                        "id": item["id"],
                        "values": item["embedding"],
                        "metadata": {
                            "text": item["text"],
                            "start_char": item.get("start_char"),
                            "end_char": item.get("end_char"),
                            "char_count": item.get("char_count")
                        }
                    }
                    for item in batch
                ]
                
                # Upsert batch to Pinecone
                self.index.upsert(vectors=batch_vectors, namespace=self.namespace)
                print(f"Upserted batch {i//batch_size + 1} ({len(batch)} vectors)")
            
            return True
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
            return False
    
    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for similar items.
        
        Args:
            query_embedding: Embedding vector for the query
            top_k: Number of similar items to retrieve
            
        Returns:
            List of similar items with metadata
        """
        try:
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=top_k,
                include_values=False,
                include_metadata=True
            )
            
            return [
                {
                    "id": match["id"],
                    "score": match["score"],
                    "text": match["metadata"].get("text", ""),
                    "start_char": match["metadata"].get("start_char"),
                    "end_char": match["metadata"].get("end_char"),
                    "char_count": match["metadata"].get("char_count")
                }
                for match in results["matches"]
            ]
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []
    
    def delete_all(self) -> bool:
        """
        Delete all vectors in the namespace.
        
        Returns:
            bool: Success status
        """
        try:
            self.index.delete(delete_all=True, namespace=self.namespace)
            print(f"Deleted all vectors in namespace: {self.namespace}")
            return True
        except Exception as e:
            print(f"Error deleting vectors: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the index.
        
        Returns:
            Dict with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            print(f"Error getting index stats: {e}")
            return {}
