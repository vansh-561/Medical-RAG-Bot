"""
Utility functions for medical RAG bot.
"""

import os
import time
import tiktoken
from dotenv import load_dotenv
from typing import List, Dict, Any#, Union, Optional

def load_environment_variables():
    """Load environment variables from .env file"""
    load_dotenv()
    required_vars = [
        "PINECONE_API_KEY", 
        "GOOGLE_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("Environment variables loaded successfully")

def get_config():
    """Get configuration from environment variables"""
    return {
        "chunk_size": int(os.environ.get("CHUNK_SIZE", 500)),
        "chunk_overlap": int(os.environ.get("CHUNK_OVERLAP", 50)),
        "embedding_model": os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "index_name": os.environ.get("INDEX_NAME", "medical-book"),
        "namespace": os.environ.get("NAMESPACE", "med-textbook"),
    }

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: Text to count tokens for
        
    Returns:
        int: Number of tokens
    """
    try:
        # Using tiktoken with cl100k_base encoding (used by many modern models)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback: rough estimate (not as accurate)
        return len(text) // 4

def truncate_text_to_token_limit(text: str, max_tokens: int = 8000) -> str:
    """
    Truncate text to stay within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        
    Returns:
        str: Truncated text
    """
    if not text:
        return ""
    
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate tokens and decode
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def create_formatted_context(sources: List[Dict[str, Any]], max_tokens: int = 6000) -> str:
    """
    Create a well-formatted context string from source documents,
    ensuring it stays within token limits.
    
    Args:
        sources: List of source documents
        max_tokens: Maximum tokens for the context
        
    Returns:
        str: Formatted context
    """
    if not sources:
        return ""
    
    # Start building context with source information
    chunks = []
    for i, source in enumerate(sources):
        chunk = f"Document {i+1} (relevance: {source['score']:.2f}):\n{source['text']}\n\n"
        chunks.append(chunk)
    
    # Join chunks and ensure within token limit
    context = "".join(chunks)
    return truncate_text_to_token_limit(context, max_tokens)

def format_elapsed_time(start_time: float) -> str:
    """
    Format elapsed time in seconds.
    
    Args:
        start_time: Start time from time.time()
        
    Returns:
        str: Formatted elapsed time
    """
    elapsed = time.time() - start_time
    if elapsed < 1:
        return f"{elapsed*1000:.0f} ms"
    elif elapsed < 60:
        return f"{elapsed:.2f} seconds"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes} min {seconds:.1f} sec"
