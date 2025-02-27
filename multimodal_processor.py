import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import whisper
from transformers import AutoTokenizer, AutoModel
import torch

class MultimodalProcessor:
    def __init__(self):
        # Initialize models for different modalities
        self.whisper_model = None  # Lazy loading
        self.code_model = None     # Lazy loading
        self.clip_model = None     # Lazy loading
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text and images from PDF files"""
        text = ""
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            # Extract text
            text += page.get_text("text")
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Save image temporarily
                img_filename = f"temp_img_{page_num}_{img_index}.png"
                with open(img_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                
                # Perform OCR on image (with error handling)
                try:
                    image = Image.open(img_filename)
                    image_text = pytesseract.image_to_string(image)
                    if image_text.strip():
                        text += f"\n[Image Text: {image_text}]\n"
                except pytesseract.TesseractNotFoundError:
                    text += f"\n[Image present but OCR unavailable - Tesseract not installed]\n"
                except Exception as e:
                    text += f"\n[Error processing image: {str(e)}]\n"
                
                # Add image path to list for later processing
                images.append(img_filename)
        
        return {"text": text, "images": images}
    
    def process_image(self, image_path):
        """Process image and extract text via OCR"""
        if not os.path.exists(image_path):
            return {"text": "", "error": "Image file not found"}
        
        image = Image.open(image_path)
        
        try:
            text = pytesseract.image_to_string(image)
        except pytesseract.TesseractNotFoundError:
            text = "[OCR unavailable - Tesseract not installed]"
            # Re-raise with more helpful message
            raise Exception("Tesseract OCR is not installed or not in PATH. Please install Tesseract to enable OCR functionality.")
        except Exception as e:
            text = f"[Error in OCR: {str(e)}]"
        
        return {"text": text, "image_path": image_path}
    
    def audio_to_text(self, audio_path):
        """Convert audio to text using Whisper"""
        try:
            if self.whisper_model is None:
                # Load model on first use (faster-whisper or whisper)
                self.whisper_model = whisper.load_model("base")
            
            result = self.whisper_model.transcribe(audio_path)
            return {"text": result["text"], "segments": result["segments"]}
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}. Make sure whisper is properly installed.")
    
    def process_code(self, code_text, language=None):
        """Process code using CodeBERT for better embeddings"""
        try:
            if self.code_model is None:
                # Initialize CodeBERT
                self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")
                self.code_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
            
            # Get CodeBERT embeddings
            inputs = self.code_tokenizer(code_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.code_model(**inputs)
            
            # Get code representation from last hidden state
            code_embedding = outputs.last_hidden_state.mean(dim=1).numpy()
            
            return {
                "text": code_text,
                "language": language,
                "embedding": code_embedding
            }
        except Exception as e:
            # Fallback to simple embedding if model fails
            return {
                "text": code_text,
                "language": language,
                "embedding": np.random.rand(768).reshape(1, -1).astype('float32')  # Dummy embedding
            }
    
    def cleanup_temp_files(self, file_list):
        """Clean up temporary files created during processing"""
        for file_path in file_list:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")