import os
import sys
import logging
from pathlib import Path

# Create necessary directories
os.makedirs("temp", exist_ok=True)
os.makedirs("vector_db", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import our modules
try:
    from multimodal_processor import MultimodalProcessor
    from multimodal_vectordb import MultimodalVectorDB
    from multimodal_rag import MultimodalRAG
    from intelligent_agent import IntelligentAgent
    logger.info("Successfully imported custom modules")
except ImportError as e:
    logger.error(f"Error importing modules: {e}")
    sys.exit(1)

def setup_environment():
    """Initialize the environment and required components"""
    logger.info("Setting up environment...")
    
    # Check for required directories
    for directory in ["temp", "vector_db"]:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
    
    # Check for Ollama
    try:
        import ollama
        logger.info("Ollama package found")
    except ImportError:
        logger.error("Ollama package not found. Please install with: pip install ollama")
        sys.exit(1)
    
    # Initialize components
    try:
        processor = MultimodalProcessor()
        logger.info("Initialized MultimodalProcessor")
        
        vector_db = MultimodalVectorDB(base_path="vector_db")
        logger.info("Initialized MultimodalVectorDB")
        
        rag_system = MultimodalRAG(vector_db)
        logger.info("Initialized MultimodalRAG")
        
        agent = IntelligentAgent(rag_system, vector_db, processor)
        logger.info("Initialized IntelligentAgent")
        
        return {
            "processor": processor,
            "vector_db": vector_db,
            "rag_system": rag_system,
            "agent": agent
        }
    
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

def run_streamlit():
    """Run the Streamlit app"""
    logger.info("Starting Streamlit app...")
    
    # Build the command to run the Streamlit app
    cmd = f"streamlit run streamlit_ui.py"
    
    # Execute the command
    logger.info(f"Executing: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    try:
        # Setup environment
        components = setup_environment()
        logger.info("Environment setup complete")
        
        # Run Streamlit app
        run_streamlit()
    
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
