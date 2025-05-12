"""
PDF processing module for medical RAG bot.
Handles PDF text extraction and chunking.
"""

import os
from typing import List, Dict, Any
from pypdf import PdfReader
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        
        print(f"Extracting text from {len(reader.pages)} pages...")
        for i, page in enumerate(tqdm(reader.pages)):
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
        
        print(f"Extracted {len(text)} characters from PDF.")
        return text
    
    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into chunks with specified overlap.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List[Dict]: List of chunks with metadata
        """
        chunks = []
        start = 0
        end = min(self.chunk_size, len(text))
        
        print(f"Chunking text with size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        chunk_id = 0
        while start < len(text):
            # Create chunk
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "char_count": len(chunk_text)
            }
            
            chunks.append(chunk)
            
            # Move to next chunk
            start = end - self.chunk_overlap
            end = min(start + self.chunk_size, len(text))
            chunk_id += 1
        
        print(f"Created {len(chunks)} chunks.")
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF and split into chunks.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List[Dict]: List of text chunks with metadata
        """
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.chunk_text(text)
        return chunks


def get_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Utility function to process a PDF file and return chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        
    Returns:
        List[Dict]: List of text chunks with metadata
    """
    processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return processor.process_pdf(pdf_path)