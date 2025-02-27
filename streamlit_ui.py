import streamlit as st
import time
import os
import base64
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from multimodal_processor import MultimodalProcessor
from multimodal_vectordb import MultimodalVectorDB
from multimodal_rag import MultimodalRAG
from intelligent_agent import IntelligentAgent

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = MultimodalProcessor()

if 'vector_db' not in st.session_state:
    st.session_state.vector_db = MultimodalVectorDB(base_path="vector_db")

if 'rag' not in st.session_state:
    st.session_state.rag = MultimodalRAG(st.session_state.vector_db)

if 'agent' not in st.session_state:
    st.session_state.agent = IntelligentAgent(
        st.session_state.rag,
        st.session_state.vector_db,
        st.session_state.processor
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def get_file_extension(file):
    """Get file extension from uploaded file."""
    return os.path.splitext(file.name)[1].lower()

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp directory and return path."""
    file_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def reset_file_state():
    """Reset file-related session state variables"""
    if 'current_file' in st.session_state:
        del st.session_state.current_file
    if 'last_uploaded_file' in st.session_state:
        del st.session_state.last_uploaded_file
    if 'file_processed' in st.session_state:
        del st.session_state.file_processed

def display_file_preview(file_path):
    """Display a preview of the uploaded file."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
        # Fixed: Use use_container_width instead of use_column_width
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
    
    elif file_extension == '.pdf':
        st.info(f"PDF file uploaded: {os.path.basename(file_path)}")
    
    elif file_extension in ['.mp3', '.wav', '.ogg']:
        st.audio(file_path, format=f'audio/{file_extension[1:]}')
    
    elif file_extension in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.txt']:
        with open(file_path, 'r') as f:
            code = f.read()
        st.code(code, language=file_extension[1:])
    
    else:
        st.info(f"File uploaded: {os.path.basename(file_path)}")

# Define the main UI layout
st.set_page_config(
    page_title="Multimodal RAG & Agent System",
    page_icon="🤖",
    layout="wide"
)
def cleanup_temp_files(directory="temp", max_age_hours=24):
    """Clean up temporary files older than the specified maximum age"""
    try:
        if not os.path.exists(directory):
            return
            
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            # Skip if not a file
            if not os.path.isfile(file_path):
                continue
                
            # Check file age
            file_age = current_time - os.path.getmtime(file_path)
            
            # Remove if older than max age
            if file_age > max_age_seconds:
                try:
                    os.remove(file_path)
                    print(f"Removed old temp file: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    except Exception as e:
        print(f"Error cleaning temp files: {e}")

def main():
    st.title("🤖 Multimodal RAG & Agent System")
    st.subheader("Process text, images, audio, and code with Ollama & LangChain")
    
    # Sidebar for file uploads and processing
    with st.sidebar:
        st.header("📁 Upload Files")
        
        # File uploader for different file types
        uploaded_file = st.file_uploader(
            "Upload a file (PDF, image, audio, code)",
            type=["pdf", "jpg", "jpeg", "png", "gif", "mp3", "wav", "ogg", "py", "js", "java", "cpp", "txt"]
        )
        
        # Process uploaded file
        if uploaded_file:
            # Check if it's a new file
            is_new_file = ('last_uploaded_file' not in st.session_state or 
                        st.session_state.last_uploaded_file != uploaded_file.name)
            
            if is_new_file:
                # Reset state for new file
                reset_file_state()
                st.session_state.last_uploaded_file = uploaded_file.name
                st.session_state.file_processed = False
            
            st.write(f"Processing: {uploaded_file.name}")
            
            # Save uploaded file
            file_path = save_uploaded_file(uploaded_file)
            st.session_state.current_file = file_path
            
            # Display file preview
            st.subheader("File Preview")
            display_file_preview(file_path)
            
            if is_new_file or not st.session_state.file_processed:
                # Process file based on type
                file_extension = get_file_extension(uploaded_file)
                
                if file_extension in ['.pdf']:
                    with st.spinner("Processing PDF..."):
                        result = st.session_state.processor.extract_text_from_pdf(file_path)
                        
                        # Index text content
                        num_chunks = st.session_state.vector_db.index_text(result["text"])
                        
                        # Index images if found
                        if result["images"]:
                            st.session_state.vector_db.index_images(result["images"])
                            
                        st.success(f"PDF processed! Indexed {num_chunks} text chunks and {len(result['images'])} images.")
                
                elif file_extension in ['.jpg', '.jpeg', '.png', '.gif']:
                    with st.spinner("Processing image..."):
                        try:
                            # Process image
                            img_data = st.session_state.processor.process_image(file_path)
                            
                            # Index image
                            st.session_state.vector_db.index_images([file_path], [img_data["text"]])
                            
                            # Also index OCR text
                            if img_data["text"]:
                                st.session_state.vector_db.index_text(img_data["text"])
                            
                            st.success(f"Image processed and indexed!")
                        except Exception as e:
                            st.error(f"Error processing image: {str(e)}")
                            st.info("Try installing Tesseract OCR or check if it's in your PATH.")
                
                elif file_extension in ['.mp3', '.wav', '.ogg']:
                    with st.spinner("Transcribing audio..."):
                        try:
                            # Process audio
                            audio_data = st.session_state.processor.audio_to_text(file_path)
                            
                            # Index transcription
                            st.session_state.vector_db.index_text(audio_data["text"])
                            
                            st.success(f"Audio transcribed and indexed!")
                            st.write(f"**Transcription**: {audio_data['text'][:200]}...")
                        except Exception as e:
                            st.error(f"Error processing audio: {str(e)}")
                
                elif file_extension in ['.py', '.js', '.java', '.cpp', '.txt']:
                    with st.spinner("Processing code..."):
                        # Read the file content
                        with open(file_path, 'r') as f:
                            code_text = f.read()
                        
                        # Process code
                        code_language = file_extension[1:]  # Remove the dot
                        code_data = st.session_state.processor.process_code(code_text, code_language)
                        
                        # Index code
                        st.session_state.vector_db.index_code([code_data["embedding"]], [code_text], [code_language])
                        
                        # Also index as text
                        st.session_state.vector_db.index_text(code_text)
                        
                        st.success(f"Code processed and indexed!")
            st.session_state.file_processed = True
        # Add a separator
        st.markdown("---")
        
        # Add a clear button
        if st.button("🧹 Clear History & Cache", use_container_width=True):
            # Clear chat history
            st.session_state.chat_history = []
            
            # Reset file state
            if 'current_file' in st.session_state:
                del st.session_state.current_file
            if 'last_uploaded_file' in st.session_state:
                del st.session_state.last_uploaded_file
            if 'file_processed' in st.session_state:
                del st.session_state.file_processed
                
            # Clear the related content display
            if 'last_query' in st.session_state:
                del st.session_state.last_query
                
            # Force refresh
            st.rerun()
        
    # Main chat area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat")
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.write(f"**You**: {message['content']}")
            else:
                st.write(f"**Assistant**: {message['content']}")
        
        # Query input
        query = st.text_input("Enter your query:", key="query_input")
        
        col_a, col_b, col_c, col_d = st.columns(4)
        
        with col_a:
            text_button = st.button("🔍 Search", use_container_width=True)
        
        with col_b:
            agent_button = st.button("🤖 Agent", use_container_width=True)
        
        with col_c:
            summary_button = st.button("📝 Summarize", use_container_width=True)
        
        with col_d:
            code_button = st.button("💻 Generate Code", use_container_width=True)
        
        # Process query
        if query and (text_button or agent_button or summary_button or code_button):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Get current file path if available
            current_file = None
            if 'uploaded_file' in locals() and uploaded_file:
                current_file = save_uploaded_file(uploaded_file)
            
            # Process query based on button clicked
            with st.spinner("Processing..."):
                if text_button:
                    response = st.session_state.rag.query_text(query)
                    query_type = "text"
                
                elif agent_button:
                    response = st.session_state.agent.run(query, file_path=current_file)
                    query_type = "agent"
                
                elif summary_button:
                    response = st.session_state.agent.generate_summary(query)
                    query_type = "summary"
                
                elif code_button:
                    response = st.session_state.agent.generate_code_snippet(query)
                    query_type = "code"
            
            # Add response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            
            # Fixed: Use st.rerun() instead of st.experimental_rerun()
            st.rerun()
    
    # Replace the code in your streamlit_ui.py that tries to display code snippets
# with this more robust version:

    with col2:
        st.header("🔍 Search Results")
        
        # Recent query panel
        if st.session_state.chat_history:
            last_query = next((msg["content"] for msg in reversed(st.session_state.chat_history) 
                            if msg["role"] == "user"), None)
            
            if last_query:
                st.subheader("Related Content")
                
                # Get relevant documents
                try:
                    docs = st.session_state.vector_db.search_text(last_query, k=3)
                    
                    for i, doc in enumerate(docs):
                        with st.expander(f"Document {i+1}"):
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                except Exception as e:
                    st.info("No text documents found in the database.")
                
                # Try to get related images if any
                try:
                    # Check if search_images method exists
                    if hasattr(st.session_state.vector_db, "search_images"):
                        images = st.session_state.vector_db.search_images(query=last_query, k=2)
                        if images and len(images) > 0:
                            st.subheader("Related Images")
                            for img in images:
                                if "image_path" in img and os.path.exists(img["image_path"]):
                                    st.image(img["image_path"], width=200)
                except Exception as e:
                    pass  # Silently fail if images can't be displayed
                
                # Simple message about code
                st.info("Code search functionality is not yet available. Upload code files to enable this feature.")
if __name__ == "__main__":
    # Clean up old temporary files
    cleanup_temp_files()
    main()