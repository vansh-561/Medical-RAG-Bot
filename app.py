"""
Streamlit app for Medical RAG Chatbot.
"""

# Ensure 'src' is in the Python path
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import time
import streamlit as st

#from med_rag_bot.pdf_processor import PDFProcessor
from src.med_rag_bot.embeddings import get_embedding_generator
from src.med_rag_bot.vector_store import VectorStore
from src.med_rag_bot.llm import GeminiLLM
from src.med_rag_bot.rag_engine import RAGEngine
from src.med_rag_bot.utils import load_environment_variables, get_config, format_elapsed_time



from dotenv import load_dotenv
load_dotenv()


# Set page configuration
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .source-box {
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 5px;
        font-size: 0.9em;
    }
    .response-time {
        color: #888;
        font-size: 0.8em;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
@st.cache_resource
def initialize_environment():
    """Initialize environment variables and configuration"""
    try:
        load_environment_variables()
        return True
    except Exception as e:
        st.error(f"Error loading environment: {e}")
        return False

# Initialize RAG components
@st.cache_resource
def initialize_rag_engine(embedding_type="sentence-transformer"):
    """Initialize and return the RAG engine"""
    try:
        # Get configuration
        config = get_config()
        
        # Initialize embedding generator
        if embedding_type == "sentence-transformer":
            embedding_generator = get_embedding_generator(
                embedding_type="sentence-transformer", 
                model_name=config["embedding_model"]
            )
        else:
            embedding_generator = get_embedding_generator(embedding_type="gemini")
        
        # Initialize vector store - removed environment parameter for Pinecone 3.0.0+
        vector_store = VectorStore(
            index_name=config["index_name"],
            namespace=config["namespace"],
            dimension=embedding_generator.dimension
        )
        
        # Initialize LLM with Gemini 2.0 Flash
        llm = GeminiLLM(model_name="gemini-2.0-flash")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            embedding_generator=embedding_generator,
            vector_store=vector_store,
            llm=llm,
            top_k=5,
            similarity_threshold=0.70
        )
        
        return rag_engine
    except Exception as e:
        st.error(f"Error initializing RAG engine: {e}")
        return None

def process_pdf_upload(uploaded_file, chunk_size, chunk_overlap):
    """Process uploaded PDF file"""
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_pdf_path = f"temp_{uploaded_file.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info(f"Processing PDF: {uploaded_file.name}...")
            
            # Ingest PDF using RAG engine
            start_time = time.time()
            success = rag_engine.ingest_pdf(
                pdf_path=temp_pdf_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            if success:
                elapsed = format_elapsed_time(start_time)
                st.success(f"✅ PDF processed successfully in {elapsed}!")
                
                # Get index stats
                stats = rag_engine.vector_store.get_stats()
                if stats:
                    # Updated for Pinecone 3.0.0 response format
                    vector_count = stats.get("namespaces", {}).get(rag_engine.vector_store.namespace, {}).get("vector_count", 0)
                    st.info(f"📊 Index now contains {vector_count} vectors.")
            else:
                st.error("❌ Failed to process PDF.")
        
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)

def process_user_query(query):
    """Process user query through the RAG engine"""
    if not query:
        return
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response with spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("🤔 Thinking...")
        
        # Process query
        start_time = time.time()
        try:
            result = rag_engine.query(query)
            elapsed = format_elapsed_time(start_time)
            
            # Display response
            message_placeholder.empty()
            st.markdown(result["response"])
            st.markdown(f"<div class='response-time'>Response time: {elapsed}</div>", unsafe_allow_html=True)
            
            # Display sources if available
            if result.get("has_context", False) and result.get("sources"):
                with st.expander("View Sources", expanded=False):
                    st.markdown("### Retrieved passages from the medical textbook")
                    for i, source in enumerate(result["sources"]):
                        st.markdown(f"**Passage {i+1}** (relevance score: {source['score']:.2f})")
                        st.markdown(
                            f"<div class='source-box'>{source['text']}</div>", 
                            unsafe_allow_html=True
                        )
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": result["response"],
                "metadata": {
                    "has_context": result.get("has_context", False),
                    "is_out_of_scope": result.get("is_out_of_scope", True),
                    "sources_count": len(result.get("sources", [])),
                    "response_time": elapsed
                }
            })
            
        except Exception as e:
            message_placeholder.error(f"Error: {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I'm sorry, I encountered an error: {e}",
                "metadata": {"error": str(e)}
            })

def main():
    """Main application function"""
    # Set up session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Application header
    st.title("🏥 Medical RAG Chatbot")
    st.markdown("""
    This chatbot answers medical questions based on the information in an uploaded medical textbook.
    Upload a medical PDF textbook to get started, then ask questions related to its content.
    """)
    
    # Initialize environment
    env_loaded = initialize_environment()
    if not env_loaded:
        st.error("Failed to initialize environment. Please check your .env file.")
        return
    
    # Initialize RAG engine
    global rag_engine
    rag_engine = initialize_rag_engine(embedding_type="sentence-transformer")
    if rag_engine is None:
        st.error("Failed to initialize RAG engine. Please check your configuration.")
        return
    
    # Sidebar with PDF upload and settings
    with st.sidebar:
        st.header("📚 Upload Medical Textbook")
        uploaded_file = st.file_uploader("Upload a medical PDF textbook", type=["pdf"])
        
        st.header("⚙️ Settings")
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=1000, value=500, step=50,
                              help="Size of text chunks for processing (in characters)")
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=200, value=50, step=10,
                                 help="Overlap between chunks (in characters)")
        
        if st.button("Process PDF", disabled=uploaded_file is None):
            process_pdf_upload(uploaded_file, chunk_size, chunk_overlap)
        
        st.markdown("---")
        st.markdown("### 📊 Model Information")
        st.markdown(f"""
        - **Embedding Model:** Sentence Transformer ({get_config()['embedding_model']})
        - **LLM:** Gemini 2.0 Flash
        - **Vector DB:** Pinecone
        - **PDF Library:** PyPDF {os.environ.get('PYPDF_VERSION', '5.5.0')}
        """)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display metadata for assistant messages if available
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                if "response_time" in metadata:
                    st.markdown(
                        f"<div class='response-time'>Response time: {metadata['response_time']}</div>", 
                        unsafe_allow_html=True
                    )
    
    # Chat input
    user_query = st.chat_input("Ask a medical question...")
    if user_query:
        process_user_query(user_query)

if __name__ == "__main__":
    main()
