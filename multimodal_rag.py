import os
import ollama
import json
from langchain_community.llms import Ollama  # Updated import
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class MultimodalRAG:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Initialize different LLMs
        self.text_llm = self._init_text_llm()
        self.code_llm = self._init_code_llm()
        self.vision_llm = None  # Will initialize on demand
        
        # Create QA chain for text
        self.qa_chain = self._create_qa_chain()
    
    def _init_text_llm(self):
        """Initialize the text LLM with optimizations"""
        # Update to match the current Ollama API in langchain
        return Ollama(
            model="deepseek-r1:14b",
            # Use model parameters in a compatible way - these might need adjustment
            temperature=0.2,
            # Note: Removed additional_kwargs that were causing errors
        )
    
    def _init_code_llm(self):
        """Initialize CodeLlama for code-related queries"""
        return Ollama(
            model="codellama:7b-instruct",
            temperature=0.2,
            # Note: Removed additional_kwargs that were causing errors
        )
    
    def _init_vision_llm(self):
        """Initialize DeepSeek-VL for vision-related queries"""
        # Note: This is a placeholder as Ollama might not directly support DeepSeek-VL
        # In practice, you might need to use an API or direct model implementation
        return None  # Replace with actual implementation
    
    def _create_qa_chain(self):
        """Create QA chain for text queries"""
        prompt_template = """
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.text_llm,
            chain_type="stuff",
            retriever=self.vector_db.text_db.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return chain
    
    def query_text(self, query):
        """Query the text database and generate response"""
        try:
            return self.qa_chain.run(query)
        except Exception as e:
            print(f"Error in query_text: {str(e)}")
            # Fallback to direct LLM call
            return self.text_llm.predict(f"Please answer this question: {query}")
    
    def query_code(self, query):
        """Query the code database and generate code-related response"""
        try:
            # Get relevant code snippets
            code_results = self.vector_db.search_code(query, k=3)
            
            # Prepare context
            code_context = ""
            if code_results and len(code_results) > 0:
                code_context = "\n\n".join([f"```{r.get('language', '')}\n{r.get('code', 'No code available')}\n```" for r in code_results])
            
            # Create prompt
            prompt = f"""
            I need help with a code-related question. Here are some relevant code snippets:
            
            {code_context if code_context else "No code snippets available."}
            
            Question: {query}
            
            Please provide a detailed answer with explanation and if applicable, write correct and improved code.
            """
            
            # Generate response using Code LLM
            response = self.code_llm.predict(prompt)
            
            return {
                "response": response,
                "references": code_results
            }
        except Exception as e:
            print(f"Error in query_code: {str(e)}")
            return {
                "response": f"Error processing code query: {str(e)}",
                "references": []
            }
    
    def query_image(self, query, image_path=None):
        """Query involving images"""
        try:
            # For now, we'll use a simpler approach without DeepSeek-VL
            # Get relevant images
            image_results = self.vector_db.search_images(query=query, k=3)
            
            # Prepare context with OCR text from images
            image_context = ""
            if image_results and len(image_results) > 0:
                image_context = "\n\n".join([f"Image content: {r.get('text', 'No text available')}" for r in image_results])
            
            # Create prompt
            prompt = f"""
            I need help with a question related to images. Here is text extracted from relevant images:
            
            {image_context if image_context else "No image text available."}
            
            Question: {query}
            
            Please provide a detailed answer based on the image content.
            """
            
            # Generate response
            response = self.text_llm.predict(prompt)
            
            return {
                "response": response,
                "references": image_results
            }
        except Exception as e:
            print(f"Error in query_image: {str(e)}")
            return {
                "response": f"Error processing image query: {str(e)}",
                "references": []
            }
    
    def multimodal_query(self, query, query_type=None, file_path=None):
        """Unified query interface that automatically detects and routes to appropriate handler"""
        try:
            # Auto-detect query type if not specified
            if not query_type:
                if "code" in query.lower() or "program" in query.lower() or "function" in query.lower():
                    query_type = "code"
                elif file_path and file_path.lower().endswith(('.jpg', '.png', '.jpeg', '.gif')):
                    query_type = "image"
                else:
                    query_type = "text"
            
            # Route to appropriate handler
            if query_type == "code":
                return self.query_code(query)
            elif query_type == "image":
                return self.query_image(query, file_path)
            else:
                return {"response": self.query_text(query)}
        except Exception as e:
            print(f"Error in multimodal_query: {str(e)}")
            return {"response": f"Error processing query: {str(e)}"}