from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.llms import Ollama  # Updated import
import json

class IntelligentAgent:
    def __init__(self, rag_system, vector_db, multimodal_processor):
        self.rag = rag_system
        self.vector_db = vector_db
        self.processor = multimodal_processor
        
        # Initialize base LLM for the agent
        self.llm = Ollama(
            model="deepseek-r1:14b",
            temperature=0.2
            # Removed additional_kwargs to match current API
        )
        
        # Create tools
        self.tools = self._create_tools()
        
        # Initialize the agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True
        )
    
    def _create_tools(self):
        """Create various tools for the agent"""
        tools = [
            Tool(
                name="TextSearch",
                func=lambda q: json.dumps({"results": [doc.page_content for doc in self.vector_db.search_text(q)]}),
                description="Search for text documents based on a query. Use this for general information retrieval."
            ),
            Tool(
                name="CodeSearch",
                func=lambda q: json.dumps({"results": self.vector_db.search_code(q)}),
                description="Search for code snippets based on a query. Use this for programming-related questions."
            ),
            Tool(
                name="ImageSearch",
                func=lambda q: json.dumps({"results": self.vector_db.search_images(query=q)}),
                description="Search for images based on a text query. Use this for image-related questions."
            ),
            Tool(
                name="GenerateSummary",
                func=lambda q: self.generate_summary(q),
                description="Generate a summary of documents related to the query."
            ),
            Tool(
                name="GenerateVisualization",
                func=lambda q: self.generate_visualization_code(q),
                description="Generate code for data visualization based on the query."
            ),
            Tool(
                name="TranslateContent",
                func=lambda q: self.translate_content(q),
                description="Translate content into different languages."
            ),
            Tool(
                name="GenerateCodeSnippet",
                func=lambda q: self.generate_code_snippet(q),
                description="Generate code snippets based on the query."
            ),
            Tool(
                name="AnswerQuestion",
                func=lambda q: self.rag.multimodal_query(q)["response"],
                description="Answer questions using the RAG system. Use this as a default for general queries."
            )
        ]
        return tools
    
    def generate_summary(self, query):
        """Generate a summary of documents related to the query"""
        # Get relevant documents
        docs = self.vector_db.search_text(query, k=5)
        
        # Create a prompt for summarization
        text_content = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""
        Please provide a concise summary of the following documents related to "{query}":
        
        {text_content}
        
        Summary:
        """
        
        # Generate summary
        summary = self.llm.predict(prompt)
        return summary
    
    def generate_visualization_code(self, query):
        """Generate code for data visualization"""
        prompt = f"""
        Write Python code to create a visualization based on the query: "{query}"
        
        Use matplotlib or seaborn. The code should be complete and ready to run.
        Include code to generate sample data if needed.
        
        ```python
        # Your visualization code here
        ```
        """
        
        # Use Code LLM for better code generation
        code = self.rag.code_llm.predict(prompt)
        return code
    
    def translate_content(self, query):
        """Translate content into different languages"""
        # Parse the query to determine content and target language
        # Example query format: "Translate to French: Hello, how are you?"
        
        prompt = f"""
        Translate the following content as requested in the query:
        
        Query: {query}
        
        Provide the translation:
        """
        
        translation = self.llm.predict(prompt)
        return translation
    
    def generate_code_snippet(self, query):
        """Generate code snippets based on the query"""
        prompt = f"""
        Write code to solve the following problem:
        
        {query}
        
        Provide a clean, efficient solution with comments explaining the key parts.
        """
        
        # Use Code LLM for better code generation
        code = self.rag.code_llm.predict(prompt)
        return code
    
    def run(self, query, query_type=None, file_path=None):
        """Process a query using the agent system"""
        # If query type is explicitly provided, use specialized handlers
        if query_type:
            if query_type == "code":
                return self.generate_code_snippet(query)
            elif query_type == "summary":
                return self.generate_summary(query)
            elif query_type == "visualization":
                return self.generate_visualization_code(query)
            elif query_type == "translation":
                return self.translate_content(query)
        
        # For files, process them first
        if file_path:
            file_extension = file_path.split('.')[-1].lower()
            
            # Process based on file type
            if file_extension in ['jpg', 'jpeg', 'png', 'gif']:
                # Process image
                img_data = self.processor.process_image(file_path)
                # Include OCR text in the query context
                query = f"Image content: {img_data['text']}\n\nQuery: {query}"
            
            elif file_extension in ['mp3', 'wav', 'ogg']:
                # Process audio
                audio_data = self.processor.audio_to_text(file_path)
                # Include transcription in the query context
                query = f"Audio transcription: {audio_data['text']}\n\nQuery: {query}"
            
            elif file_extension == 'pdf':
                # Process PDF (already handled in the main application)
                pass
        
        # Let the agent decide which tools to use
        response = self.agent.run(query)
        
        return response