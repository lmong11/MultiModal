import os
import faiss
import numpy as np
import pickle
import torch
from PIL import Image
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS  # Updated import
from transformers import CLIPProcessor, CLIPModel

class MultimodalVectorDB:
    def __init__(self, base_path="vector_db"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
        # Text embeddings
        self.text_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize different vector stores for different modalities
        self.text_db_path = os.path.join(base_path, "text_faiss")
        self.image_db_path = os.path.join(base_path, "image_faiss")
        self.code_db_path = os.path.join(base_path, "code_faiss")
        
        # Create or load vector stores
        self.text_db = self._init_text_db()
        self.image_db = self._init_image_db()
        self.code_db = self._init_code_db()
        
        # Initialize CLIP for image processing (lazy loading)
        self.clip_model = None
        self.clip_processor = None
    
    def _init_text_db(self):
        """Initialize or load text vector database"""
        if os.path.exists(self.text_db_path):
            # Add allow_dangerous_deserialization=True to fix the pickle security issue
            return FAISS.load_local(
                self.text_db_path, 
                self.text_embeddings, 
                allow_dangerous_deserialization=True
            )
        return FAISS.from_texts(["initialization"], self.text_embeddings)
    
    def _init_image_db(self):
        """Initialize or load image vector database"""
        # Check if image database exists
        if os.path.exists(self.image_db_path):
            # Load existing database
            index = faiss.read_index(os.path.join(self.image_db_path, "index.faiss"))
            with open(os.path.join(self.image_db_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
            return {"index": index, "metadata": metadata}
        
        # Initialize new image database
        os.makedirs(self.image_db_path, exist_ok=True)
        # CLIP embeddings are 512-dimensional
        index = faiss.IndexFlatL2(512)
        metadata = {"paths": [], "texts": []}
        
        # Save initial database
        faiss.write_index(index, os.path.join(self.image_db_path, "index.faiss"))
        with open(os.path.join(self.image_db_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        return {"index": index, "metadata": metadata}
    
    def _init_code_db(self):
        """Initialize or load code vector database"""
        # Similar structure to image_db
        if os.path.exists(self.code_db_path):
            index = faiss.read_index(os.path.join(self.code_db_path, "index.faiss"))
            with open(os.path.join(self.code_db_path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
            return {"index": index, "metadata": metadata}
        
        os.makedirs(self.code_db_path, exist_ok=True)
        # CodeBERT embeddings are 768-dimensional
        index = faiss.IndexFlatL2(768)
        metadata = {"code_texts": [], "languages": []}
        
        # Save initial database
        faiss.write_index(index, os.path.join(self.code_db_path, "index.faiss"))
        with open(os.path.join(self.code_db_path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        return {"index": index, "metadata": metadata}
    
    def _get_clip_model(self):
        """Lazy load the CLIP model"""
        if self.clip_model is None:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return self.clip_model, self.clip_processor
    
    def index_text(self, texts, metadatas=None):
        """Add texts to the text vector database"""
        # Split texts into chunks if necessary
        # This is a simplified version - in production, use a proper text splitter
        chunks = texts if isinstance(texts, list) else [texts]
        
        # Add to vector store
        self.text_db.add_texts(chunks, metadatas=metadatas)
        # Save
        self.text_db.save_local(self.text_db_path)
        return len(chunks)
    
    def index_images(self, image_paths, image_texts=None):
        """Add images to the image vector database"""
        model, processor = self._get_clip_model()
        
        # Process each image
        for i, img_path in enumerate(image_paths):
            try:
                # Load and process image
                image = Image.open(img_path)
                inputs = processor(images=image, return_tensors="pt")
                
                # Get image embeddings
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)
                
                # Add to index
                embedding = image_features.numpy().astype('float32')
                self.image_db["index"].add(embedding)
                
                # Add metadata
                text = image_texts[i] if image_texts and i < len(image_texts) else ""
                self.image_db["metadata"]["paths"].append(img_path)
                self.image_db["metadata"]["texts"].append(text)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        # Save index and metadata
        faiss.write_index(self.image_db["index"], os.path.join(self.image_db_path, "index.faiss"))
        with open(os.path.join(self.image_db_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.image_db["metadata"], f)
        
        return len(image_paths)
    
    def index_code(self, code_embeddings, code_texts, languages=None):
        """Add code to the code vector database"""
        # Process each code snippet
        for i, embedding in enumerate(code_embeddings):
            try:
                # Add to index
                self.code_db["index"].add(embedding.reshape(1, -1))
                
                # Add metadata
                self.code_db["metadata"]["code_texts"].append(code_texts[i])
                lang = languages[i] if languages and i < len(languages) else None
                self.code_db["metadata"]["languages"].append(lang)
                
            except Exception as e:
                print(f"Error indexing code: {e}")
        
        # Save index and metadata
        faiss.write_index(self.code_db["index"], os.path.join(self.code_db_path, "index.faiss"))
        with open(os.path.join(self.code_db_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.code_db["metadata"], f)
        
        return len(code_embeddings)
    
    def search_text(self, query, k=5):
        """Search the text database"""
        return self.text_db.similarity_search(query, k=k)
    
    def search_images(self, query=None, image_path=None, k=5):
        """Search the image database using text or image query"""
        model, processor = self._get_clip_model()
        
        # Get query embedding
        if image_path:
            # Image-to-image search
            image = Image.open(image_path)
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                query_embedding = model.get_image_features(**inputs).numpy().astype('float32')
        else:
            # Text-to-image search
            inputs = processor(text=query, return_tensors="pt")
            with torch.no_grad():
                query_embedding = model.get_text_features(**inputs).numpy().astype('float32')
        
        # Search
        D, I = self.image_db["index"].search(query_embedding, k)
        
        # Get results
        results = []
        for i in range(len(I[0])):
            idx = I[0][i]
            if idx < len(self.image_db["metadata"]["paths"]):
                results.append({
                    "image_path": self.image_db["metadata"]["paths"][idx],
                    "text": self.image_db["metadata"]["texts"][idx],
                    "distance": float(D[0][i])
                })
        
        return results
    
    # This is a patch for the search_code method in multimodal_vectordb.py

    def search_code(self, query, k=5):
        """Search for code snippets related to the query"""
        try:
            # First check if the code database is initialized
            if not hasattr(self, "code_db") or self.code_db is None:
                print("Code database not initialized")
                return []
                
            # Check if the index exists and has entries
            if "index" not in self.code_db or self.code_db["index"].ntotal == 0:
                print("Code index is empty")
                return []
                
            # Get query embedding
            query_embedding = self.text_embeddings.embed_query(query)
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            
            # Adjust k if needed
            actual_k = min(k, self.code_db["index"].ntotal)
            if actual_k == 0:
                return []
                
            # Search
            D, I = self.code_db["index"].search(query_embedding, actual_k)
            
            # Get results
            results = []
            for i in range(len(I[0])):
                idx = I[0][i]
                if idx < len(self.code_db["metadata"]["code_texts"]):
                    results.append({
                        "code": self.code_db["metadata"]["code_texts"][idx],
                        "language": self.code_db["metadata"]["languages"][idx],
                        "distance": float(D[0][i])
                    })
            
            return results
        except Exception as e:
            print(f"Error in search_code: {str(e)}")
            # Return empty list instead of raising exception
            return []
        
    def clear_previous_file_data(self):
        """Clear data related to previously processed files"""
        try:
            # Reset text database
            # Note: This is a new FAISS store, not clearing the existing one
            # If you want to keep some data, you'll need a more selective approach
            self.text_db = FAISS.from_texts(["initialization"], self.text_embeddings)
            
            # Reset image database
            if hasattr(self, "image_db") and self.image_db is not None:
                if "index" in self.image_db:
                    # Create a new empty index with the same dimension
                    dimension = 512  # CLIP embeddings are 512-dimensional
                    self.image_db["index"] = faiss.IndexFlatL2(dimension)
                
                if "metadata" in self.image_db:
                    self.image_db["metadata"] = {"paths": [], "texts": []}
            
            # Reset code database
            if hasattr(self, "code_db") and self.code_db is not None:
                if "index" in self.code_db:
                    # Create a new empty index with the same dimension
                    dimension = 768  # CodeBERT embeddings are 768-dimensional
                    self.code_db["index"] = faiss.IndexFlatL2(dimension)
                
                if "metadata" in self.code_db:
                    self.code_db["metadata"] = {"code_texts": [], "languages": []}
            
            print("Cleared previous file data from vector databases")
            return True
        except Exception as e:
            print(f"Error clearing previous file data: {str(e)}")
            return False