"""
Embeddings module for medical RAG bot.
Handles vector embedding generation using Sentence Transformers or Google's embeddings.
"""

import os
from typing import List#, Dict, Any, Union
from tqdm import tqdm
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class EmbeddingGenerator:
    """Base class for embedding generation"""
    
    def __init__(self):
        pass
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        raise NotImplementedError
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        raise NotImplementedError


class SentenceTransformerEmbeddings(EmbeddingGenerator):
    """Embedding generator using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with Sentence Transformer model.
        
        Args:
            model_name: Name of the Sentence Transformer model
        """
        super().__init__()
        print(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded with embedding dimension: {self.dimension}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts using {self.model_name}")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        # Convert numpy arrays to native Python lists
        return embeddings.tolist()
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text)
        return embedding.tolist()


class GeminiEmbeddings(EmbeddingGenerator):
    """Embedding generator using Google's Gemini embeddings API"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize with Google API key.
        
        Args:
            api_key: Google API key
        """
        super().__init__()
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini embeddings")
        
        genai.configure(api_key=self.api_key)
        
        # Use Google's text embedding model - update as needed for latest model
        self.embedding_model = "models/embedding-001"
        self.dimension = 768  # May vary by model
        print(f"Using Google's embedding model with dimension: {self.dimension}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        print(f"Generating embeddings for {len(texts)} texts using Google Gemini embeddings")
        embeddings = []
        
        for text in tqdm(texts):
            try:
                embedding_model = genai.get_embedding_model(self.embedding_model)
                result = embedding_model.embed_content(
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result.embedding)
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Add a zero vector as fallback
                embeddings.append([0.0] * self.dimension)
        
        return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            embedding_model = genai.get_embedding_model(self.embedding_model)
            result = embedding_model.embed_content(
                content=text,
                task_type="retrieval_query"
            )
            return result.embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * self.dimension


def get_embedding_generator(embedding_type: str = "sentence-transformer", 
                           model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Factory function to get the appropriate embedding generator.
    
    Args:
        embedding_type: Type of embedding generator ('sentence-transformer' or 'gemini')
        model_name: Model name for sentence-transformer
        
    Returns:
        EmbeddingGenerator instance
    """
    if embedding_type.lower() == "sentence-transformer":
        return SentenceTransformerEmbeddings(model_name=model_name)
    elif embedding_type.lower() == "gemini":
        return GeminiEmbeddings()
    else:
        raise ValueError(f"Unsupported embedding type: {embedding_type}")