# Multimodal RAG & Agent System

A comprehensive multimodal system that combines Retrieval-Augmented Generation (RAG) with intelligent agents to process text, images, audio, and code, powered by Ollama and LangChain.

![Multimodal RAG System](https://github.com/lmong11/MultiModal/assets/test.png)

## 🌟 Features

- **Support for Multiple Data Types**: Process text, images, audio, and code seamlessly
- **Multimodal Vector Database**: Store and retrieve vectors for different modalities using FAISS
- **Intelligent Agents**: Automatically select the appropriate tools based on query
- **Local LLM Support**: Powered by Ollama (DeepSeek-R1 for text, CodeLlama for code)
- **OCR Integration**: Extract text from images using Tesseract
- **Voice Transcription**: Convert audio to text using Whisper
- **Code Analysis**: Parse and analyze code using CodeBERT
- **Web-Based UI**: Easy to use Streamlit interface

## 🛠️ Installation

### Prerequisites

- Python 3.9+ 
- [Ollama](https://ollama.ai/) with DeepSeek-R1 and CodeLlama models
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for image processing)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/lmong11/MultiModal.git
   cd multimodal-rag-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Tesseract OCR:
   
   **macOS**:
   ```bash
   brew install tesseract
   ```
   
   **Ubuntu/Debian**:
   ```bash
   sudo apt-get install tesseract-ocr
   ```
   
   **Windows**:
   Download from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)

5. Install and pull Ollama models:
   ```bash
   # Install Ollama from https://ollama.ai/
   
   # Pull models
   ollama pull deepseek-r1:14b
   ollama pull codellama:7b-instruct
   ```

## 🚀 Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to `http://localhost:8501`

3. Upload files (PDFs, images, audio, or code)

4. Ask questions about the uploaded content using the chat interface

5. Choose between different processing modes:
   - 🔍 Search: Simple RAG to retrieve information
   - 🤖 Agent: Let the system select the appropriate tools
   - 📝 Summarize: Generate summaries of the content
   - 💻 Generate Code: Create code related to your query

## 🏗️ Project Structure

```
multimodal-rag-agent/
├── main.py                   # Main application entry point
├── multimodal_processor.py   # Data processing module
├── multimodal_vectordb.py    # Vector database module
├── multimodal_rag.py         # RAG system module
├── intelligent_agent.py      # Agent module
├── streamlit_ui.py           # Streamlit UI
├── vector_db/                # Storage for vector databases
├── temp/                     # Temporary file storage
└── requirements.txt          # Dependencies
```

## 🧠 How It Works

1. **Data Processing**:
   - PDF files are parsed using PyMuPDF to extract text and images
   - Images are processed with Tesseract OCR to extract text
   - Audio files are transcribed using Whisper
   - Code is analyzed using CodeBERT for better embeddings

2. **Vector Storage**:
   - Different vector stores for text, images, and code
   - FAISS is used for efficient vector search
   - Sentence Transformers for text embeddings
   - CLIP for image embeddings

3. **RAG System**:
   - Queries are routed to the appropriate handler based on content type
   - Documents are retrieved based on vector similarity
   - Results are passed to Ollama models for final response generation

4. **Agent System**:
   - Decides which tools to use based on the query
   - Dynamically selects the appropriate LLM
   - Can generate summaries, visualizations, translations, and code

## 🔧 Advanced Configuration

### Setting Tesseract Path
If Tesseract is installed but not in your PATH, set it in the code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/path/to/tesseract'
```

### Changing Ollama Models
Edit the `_init_text_llm` and `_init_code_llm` methods in `multimodal_rag.py`.

### Adjusting Vector Search Parameters
Modify the `k` parameter in search methods to control the number of results.

## 🙏 Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the framework
- [Ollama](https://ollama.ai/) for local LLM support
- [FAISS](https://github.com/facebookresearch/faiss) for vector search
- [Streamlit](https://streamlit.io/) for the web interface
- [DeepSeek](https://github.com/deepseek-ai) for the DeepSeek-R1 model
- [CodeLlama](https://github.com/facebookresearch/codellama) for code generation
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for image text extraction
- [Whisper](https://github.com/openai/whisper) for audio transcription
