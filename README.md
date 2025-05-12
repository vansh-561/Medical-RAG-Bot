# Medical RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for medical information, built with Python. The chatbot uses a medical textbook as its knowledge base and provides accurate responses to medical queries based on the content of the book.

## Features

* **PDF Processing**: Extract and chunk text from a medical textbook PDF (600+ pages)
* **Vector Embeddings**: Generate embeddings using Sentence Transformers or Google's Gemini embeddings
* **Vector Database**: Store and retrieve embeddings using Pinecone
* **RAG-based Responses**: Retrieve relevant information and generate responses using Gemini 2.0 Flash
* **User-friendly Interface**: Clean, modern UI built with Streamlit
* **Source References**: View the exact passages from the textbook used to generate responses
* **Response Timing**: Track and display response generation time

## Project Structure

```
med-rag-bot/
├── pyproject.toml   # Poetry configuration
├── README.md        # Project documentation
├── .env             # Environment variables (API keys)
├── .gitignore       # Git ignore file
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py    # PDF extraction and chunking
│   ├── embeddings.py       # Vector embedding generation
│   ├── vector_store.py     # Pinecone database operations
│   ├── llm.py              # Gemini LLM integration
│   ├── rag_engine.py       # RAG retrieval and response logic
│   └── utils.py            # Helper functions
└── app.py                  # Streamlit application
```

## Setup and Installation

1. Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/yourusername/med-rag-bot.git
cd med-rag-bot
```

2. Install Poetry (if not already installed):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install dependencies:

```bash
poetry install
```

4. Set up environment variables by creating a `.env` file:

```
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
CHUNK_SIZE=500
CHUNK_OVERLAP=50
EMBEDDING_MODEL=all-MiniLM-L6-v2
INDEX_NAME=medical-book
NAMESPACE=med-textbook
```

5. Run the application:

```bash
poetry run streamlit run app.py
```

## Usage

1. Upload a medical textbook PDF through the Streamlit interface
2. Process the PDF and generate embeddings
3. Start asking medical questions in natural language
4. View responses with source references from the textbook

## Dependencies

- Python 3.9+
- Streamlit
- PyPDF2
- Sentence Transformers
- Google Generative AI (Gemini)
- Pinecone
- python-dotenv

## License

MIT

## Disclaimer

This tool is for educational purposes only. Always consult with a healthcare professional for medical advice.
