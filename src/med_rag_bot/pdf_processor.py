"""
PDF processing module for medical RAG bot.
Handles PDF text extraction and chunking.
"""

import os
import gc
import time
from typing import List, Dict, Any
from pypdf import PdfReader
from tqdm import tqdm

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, timeout: int = 300):
        """
        Initialize PDF processor with chunking parameters.
        
        Args:
            chunk_size: Number of characters per chunk
            chunk_overlap: Number of overlapping characters between chunks
            timeout: Maximum processing time in seconds (default: 5 minutes)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.timeout = timeout
        self.start_time = None
    
    def _check_timeout(self, operation: str = "Processing") -> None:
        """Check if processing has exceeded the timeout limit"""
        if self.start_time is None:
            self.start_time = time.time()
            
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            print(f"Timeout exceeded ({self.timeout}s). Stopping processing.")
            raise TimeoutError(f"{operation} operation timed out after {self.timeout} seconds")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        self.start_time = time.time()
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        reader = PdfReader(pdf_path)
        text = ""
        
        print(f"Extracting text from {len(reader.pages)} pages...")
        for i, page in enumerate(tqdm(reader.pages)):
            self._check_timeout("Text extraction")
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
        print(f"Chunking text with size={self.chunk_size}, overlap={self.chunk_overlap}")
        
        # Pre-allocate list with estimated size to avoid resizing
        text_length = len(text)
        estimated_chunks = max(1, (text_length - self.chunk_overlap) // 
                              (self.chunk_size - self.chunk_overlap) + 1)
        print(f"Estimated number of chunks: {estimated_chunks}")
        
        # Use a more memory-efficient approach
        chunks = []
        chunk_id = 0
        start = 0
        
        progress_step = max(1, estimated_chunks // 10)  # Show progress for every 10%
        last_progress = 0
        
        # Process one chunk at a time
        while start < text_length:
            self._check_timeout("Text chunking")
            
            end = min(start + self.chunk_size, text_length)
            chunk_text = text[start:end]
            
            # Create chunk with metadata
            chunk = {
                "id": f"chunk_{chunk_id}",
                "text": chunk_text,
                "start_char": start,
                "end_char": end,
                "char_count": len(chunk_text)
            }
            
            # Add chunk to list and immediately run garbage collection if memory is tight
            chunks.append(chunk)
            
            # Progress reporting
            if chunk_id % progress_step == 0 or chunk_id == estimated_chunks - 1:
                progress_percent = min(100, int((chunk_id + 1) / estimated_chunks * 100))
                if progress_percent > last_progress: # Only print if actual progress made to avoid spam for last chunk
                    print(f"Chunking progress: {progress_percent}% ({chunk_id+1}/{estimated_chunks})")
                    last_progress = progress_percent
            
            # If this chunk reached the end of the text, it's the last one.
            if end == text_length:
                break
            
            # Move to next chunk
            start = end - self.chunk_overlap
            # Ensure start actually progresses if overlap is too large compared to chunk_size for remaining text
            if start <= chunks[-1]['start_char'] and len(chunks) > 0 : # Check against the start of the chunk just added
                # This condition prevents re-processing or moving backward if overlap is aggressive
                # For example, if end - overlap <= previous start
                print(f"Warning: Chunking overlap might stall progress. Advancing start position from {start} to {chunks[-1]['start_char'] + 1}.")
                start = chunks[-1]['start_char'] + 1 
                if start >= text_length: # Ensure we don't go past text_length due to this adjustment
                    break

            chunk_id += 1
            
            # Force garbage collection periodically
            if chunk_id % 100 == 0:
                gc.collect()
        
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
        self.start_time = time.time()
        
        try:
            # Check file size and adjust processing parameters if needed
            file_size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
            print(f"PDF file size: {file_size_mb:.2f} MB")
            
            if file_size_mb > 10:
                print(f"Large PDF detected. Reducing chunk size for better performance.")
                self.chunk_size = min(self.chunk_size, 250)
                self.chunk_overlap = min(self.chunk_overlap, 25)
            
            text = self.extract_text_from_pdf(pdf_path)
            print(f"PDF text extraction completed in {time.time() - self.start_time:.2f} seconds")
            
            # Force garbage collection after text extraction
            gc.collect()
            
            # Reset timer for chunking phase
            self.start_time = time.time()
            chunks = self.chunk_text(text)
            print(f"PDF chunking completed in {time.time() - self.start_time:.2f} seconds")
            
            return chunks
        except TimeoutError as e:
            print(f"Processing timed out: {e}")
            # Return whatever chunks we have so far if any
            if 'chunks' in locals() and chunks:
                print(f"Returning {len(chunks)} chunks processed before timeout")
                return chunks
            raise
        except MemoryError:
            print("Memory error occurred while processing PDF. Attempting to recover...")
            gc.collect()
            # If we still have the text, try again with smaller chunks
            if 'text' in locals():
                self.chunk_size = min(self.chunk_size, 250)  # Reduce chunk size
                self.chunk_overlap = min(self.chunk_overlap, 25)  # Reduce overlap
                print(f"Retrying with smaller chunks: size={self.chunk_size}, overlap={self.chunk_overlap}")
                return self.chunk_text(text)
            else:
                raise
        except Exception as e:
            print(f"Error processing PDF: {type(e).__name__}: {e}")
            raise


def get_pdf_chunks(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50, timeout: int = 300) -> List[Dict[str, Any]]:
    """
    Utility function to process a PDF file and return chunks.
    
    Args:
        pdf_path: Path to the PDF file
        chunk_size: Number of characters per chunk
        chunk_overlap: Number of overlapping characters between chunks
        timeout: Maximum processing time in seconds
        
    Returns:
        List[Dict]: List of text chunks with metadata
    """
    processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap, timeout=timeout)
    return processor.process_pdf(pdf_path)