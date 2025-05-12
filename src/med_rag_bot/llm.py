"""
LLM module for medical RAG bot.
Handles interactions with Gemini 2.0 Flash LLM.
"""

import os
#from typing import Dict, List, Any, Optional, Union
import google.generativeai as genai

class GeminiLLM:
    """Interface to Google's Gemini LLM API"""
    
    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """
        Initialize Gemini LLM.
        
        Args:
            api_key: Google API key
            model_name: Gemini model name to use
        """
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key is required for Gemini LLM")
        
        # Configure the API
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        
        # Configure generation parameters
        generation_config = genai.GenerationConfig(
            temperature=0.3,  # Low temperature for more focused, factual responses
            top_p=0.95,
            top_k=40,
            max_output_tokens=1024,
        )
        
        # Create the model
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config
        )
        
        print(f"Initialized Gemini LLM with model: {model_name}")
    
    def generate_response(self, query: str, context: str = None) -> str:
        """
        Generate a response from Gemini based on a query and optional context.
        
        Args:
            query: User's query
            context: Optional context from retrieved documents
            
        Returns:
            Generated response
        """
        try:
            if context:
                prompt = self._create_rag_prompt(query, context)
            else:
                prompt = self._create_standard_prompt(query)
            
            response = self.model.generate_content(prompt)
            
            return response.text
        except Exception as e:
            print(f"Error generating response from Gemini: {e}")
            return "I'm sorry, I encountered an error while generating a response. Please try again."
    
    def _create_standard_prompt(self, query: str) -> str:
        """
        Create a standard prompt for non-RAG queries.
        
        Args:
            query: User's query
            
        Returns:
            Formatted prompt
        """
        return f"""You are a medical information assistant based on a specific medical textbook.
        
Question: {query}

If this question is outside the scope of your medical knowledge, please respond with:
"I'm sorry, but I don't have information about that in my medical textbook."

Otherwise, provide a clear, accurate, and helpful response to the medical question."""
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """
        Create a RAG-enhanced prompt with context from retrieved documents.
        
        Args:
            query: User's query
            context: Context from retrieved documents
            
        Returns:
            Formatted prompt
        """
        return f"""You are a medical information assistant based on a specific medical textbook.
        
Below is information from the medical textbook that may be relevant to the question:

{context}

Question: {query}

Based ONLY on the information provided above, please answer the question. 
If the information provided doesn't contain the answer, respond with:
"I'm sorry, but I don't have information about that in my medical textbook."

Your response should be:
1. Accurate and based only on the provided context
2. Clear and easy to understand
3. Properly formatted for readability
4. Concise but thorough"""
