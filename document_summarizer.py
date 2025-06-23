import os
import time
import json
import argparse
import traceback 
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
# Embeddings and vector store
from sentence_transformers import SentenceTransformer
import faiss
# Groq API
from groq import Groq

# Load environment variables
load_dotenv()

class SmartDocumentProcessor:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """Initialize document processor with adaptive chunking strategy"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use RecursiveCharacterTextSplitter for better semantic boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""]
        )
        
        print(f"✅ Document Processor: Initialized with adaptive chunking")
        print(f"   Chunk size: {chunk_size} chars, Overlap: {chunk_overlap} chars")
    
    def load_document(self, file_path: str):
        """Load document with appropriate loader based on file extension"""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        print(f"📄 Loading document: {file_path}")
        
        # Use LangChain document loaders
        if file_extension == ".pdf":
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            text = " ".join([doc.page_content for doc in documents])
            print(f"   PDF document has {len(documents)} pages")
        elif file_extension == ".txt":
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            text = documents[0].page_content
        elif file_extension in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            text = documents[0].page_content
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Return as a single text string
        total_chars = len(text)
        print(f"✅ Document loaded: ~{total_chars} characters")
        return text
    
    def split_text(self, text):
        """Split text into chunks using semantic boundaries"""
        start_time = time.time()
        chunks = self.text_splitter.split_text(text)
        
        # Calculate statistics
        if chunks:
            avg_size = sum(len(chunk) for chunk in chunks) // len(chunks)
            min_size = min(len(chunk) for chunk in chunks)
            max_size = max(len(chunk) for chunk in chunks)
        else:
            avg_size, min_size, max_size = 0, 0, 0
        
        print(f"✅ Document chunked: {len(chunks)} chunks in {time.time() - start_time:.2f}s")
        print(f"   Average chunk size: {avg_size} chars (~{avg_size//4} tokens)")
        print(f"   Size range: {min_size}-{max_size} chars")
        
        return chunks

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize SentenceTransformers for vector embeddings"""
        try:
            self.model = SentenceTransformer(model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"✅ Embedding Model: {model_name} ({self.dimension}D)")
        except Exception as e:
            print(f"❌ Error initializing embedding model: {e}")
            raise
    
    def embed_chunks(self, chunks):
        """Convert chunks to vector embeddings"""
        print(f"🔄 Converting {len(chunks)} chunks to vector embeddings...")
        start_time = time.time()
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        print(f"✅ Vector embeddings created in {time.time() - start_time:.2f}s")
        return embeddings

class VectorStore:
    def __init__(self, dimension: int):
        """Initialize FAISS vector database"""
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks = []
        print(f"✅ Vector DB: FAISS IndexFlatIP initialized ({dimension}D)")
        
    def store_embeddings(self, embeddings, chunks, db_path: str = None):
        """Store embeddings and chunks in FAISS vector database"""
        faiss.normalize_L2(embeddings)  # Normalize for cosine similarity
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks
        
        if db_path:
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
            faiss.write_index(self.index, f"{db_path}.index")
            
            with open(f"{db_path}.chunks", 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Vector Storage: {len(chunks)} embeddings stored to {db_path}")
        else:
            print(f"✅ In-memory vector store created with {len(chunks)} chunks")
        
        return self
    
    def load_database(self, db_path: str):
        """Load FAISS database from disk"""
        self.index = faiss.read_index(f"{db_path}.index")
        with open(f"{db_path}.chunks", 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✅ Vector DB Loaded: {len(self.chunks)} chunks from {db_path}")
        return self
    
    def retrieve_relevant_chunks(self, query_embedding, top_k: int = 6):
        """Perform semantic retrieval for summary query"""
        faiss.normalize_L2(query_embedding.reshape(1, -1))
        scores, indices = self.index.search(query_embedding.reshape(1, -1).astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:
                results.append((self.chunks[idx], float(score)))
        
        print(f"✅ Semantic Retrieval: Retrieved top-{len(results)} most relevant chunks")
        return results

class GroqGenerator:
    def __init__(self, api_key: str, model_name: str = "llama3-70b-8192"):
        """Initialize Groq LLM client"""
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        self.processing_start_time = time.time()
        self.total_tokens_used = 0
        
        print(f"✅ LLM Initialized: {self.model_name}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters)"""
        return len(text) // 4
    
    def generate_response(self, query: str, retrieved_chunks, top_k: int = 6):
        """Generate response to query using retrieved chunks"""
        print(f"🔍 Query: '{query}'")
        print(f"   Retrieving {top_k} most relevant chunks...")
        
        # Create context from retrieved chunks
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks):
            context_parts.append(f"SECTION {i+1} (Relevance Score: {score:.3f}):\n{chunk}")
        
        context = "\n\n" + "="*60 + "\n\n".join(context_parts)
        
        # Estimate tokens
        content_tokens = self.estimate_tokens(context)
        prompt_tokens = 600  # Estimate for system + user prompt
        total_tokens = content_tokens + prompt_tokens
        
        print(f"📊 Total context: ~{content_tokens} tokens across {len(retrieved_chunks)} chunks")
        print(f"🧠 Generating response using {self.model_name}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a helpful AI assistant provides a summary based on the provided document excerpts.
                        
Focus on:
1. Technical accuracy and completeness
2. Clear hierarchical organization with headings
3. Specific implementation details and algorithms
4. Relationships between different concepts
5. Professional technical writing style

Ensure your response is coherent, fluent, and accurate while maintaining technical depth throughout."""
                    },
                    {
                        "role": "user",
                        "content": f"""Create a comprehensive technical answer about: {query}

RETRIEVED CONTEXT:
{context}

Please provide a comprehensive answer based only on the information in the context.
If the context doesn't contain relevant information to answer the question, 
say "I don't have enough information to answer this question."

Write a detailed response:"""
                    }
                ],
                max_tokens=4000,
                temperature=0.7
            )
            
            self.total_tokens_used += total_tokens + len(response.choices[0].message.content) // 4
            print("CONTEXT")
            print(context)
            print("\n" + "-"*50)
            print("✅ Response generated")
            print("\n" + "-"*50 + "\n")
            response_text = response.choices[0].message.content
            print(response_text)
            print("\n" + "-"*50)
            
            return response_text
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            traceback.print_exc()
            return f"Error generating response: {str(e)}"

class RAGPipeline:
    def __init__(self, groq_api_key: str = None, embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "llama-3.1-70b-versatile"):
        """Initialize complete RAG pipeline with all components"""
        print("🚀 Initializing RAG Document Summarization Pipeline")
        print("="*60)
        
        # Get API key from parameter or environment variable
        if groq_api_key is None:
            groq_api_key = os.getenv('GROQ_API_KEY')
            if groq_api_key is None:
                raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or pass it as a parameter.")
        
        self.doc_processor = SmartDocumentProcessor()
        self.embedding_manager = EmbeddingManager(model_name=embedding_model)
        self.vector_store = VectorStore(self.embedding_manager.dimension)
        self.generator = GroqGenerator(api_key=groq_api_key, model_name=llm_model)
        
        print("="*60)
    
    def process_document(self, filepath: str, chunk_size: int = 2000, chunk_overlap: int = 200, persist_directory: str = None):
        """Process document: load -> chunk -> embed -> store"""
        # Override document processor parameters if specified
        self.doc_processor.chunk_size = chunk_size
        self.doc_processor.chunk_overlap = chunk_overlap
        self.doc_processor.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ";", ":", " ", ""]
        )
        
        # Process document
        text = self.doc_processor.load_document(filepath)
        chunks = self.doc_processor.split_text(text)
        embeddings = self.embedding_manager.embed_chunks(chunks)
        
        # Store embeddings
        vector_store = self.vector_store.store_embeddings(embeddings, chunks, persist_directory)
        return vector_store
    
    def query_document(self, query: str, top_k: int = 6):
        """Query document using existing vector store"""
        # Embed query
        query_embedding = self.embedding_manager.embed_chunks([query])
        
        # Retrieve relevant chunks
        results = self.vector_store.retrieve_relevant_chunks(query_embedding, top_k=top_k)
        
        # Generate response
        response = self.generator.generate_response(query, results, top_k=top_k)
        
        return response

def main():
    """Main function to run the document summarizer"""
    parser = argparse.ArgumentParser(description='RAG-based Document Summarizer and Query Engine')
    
    # Required arguments
    parser.add_argument('file_path', type=str, help='Path to the document file')
    
    # Optional arguments
    parser.add_argument('--query', '-q', type=str, help='Query to ask about the document')
    parser.add_argument('--top-k', '-k', type=int, default=6, help='Number of chunks to retrieve')
    parser.add_argument('--chunk-size', type=int, default=2000, help='Size of document chunks')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Overlap between chunks')
    parser.add_argument('--model', '-m', type=str, default='llama3-70b-8192', help='LLM model to use')
    parser.add_argument('--persist-dir', '-d', type=str, help='Directory to persist vector store')
    parser.add_argument('--embedding-model', '-e', type=str, default='all-MiniLM-L6-v2', 
                        help='Embedding model to use')
    parser.add_argument('--reprocess', '-r', action='store_true', help='Force reprocessing of document')
    
    args = parser.parse_args()
    
    try:
        print(f"📄 Document: {args.file_path}")
        # Create file-specific database path
        file_basename = os.path.basename(args.file_path)
        if args.persist_dir:
            # Create the directory if it doesn't exist
            os.makedirs(args.persist_dir, exist_ok=True)
            # Use file name for the database
            db_path = os.path.join(args.persist_dir, file_basename.replace('.', '_'))
            print(f"💾 Database: {db_path}")
        else:
            db_path = None
            
        print("="*60)
        
        # Initialize the RAG pipeline
        pipeline = RAGPipeline(
            embedding_model=args.embedding_model, 
            llm_model=args.model
        )
        
        # Check if we need to process the document
        should_process = True
        if db_path and not args.reprocess:
            if os.path.exists(f"{db_path}.index") and os.path.exists(f"{db_path}.chunks"):
                try:
                    pipeline.vector_store.load_database(db_path)
                    should_process = False
                    print(f"✅ Using existing vector database for {file_basename}")
                except Exception as e:
                    print(f"⚠️ Could not load existing store: {e}")
                    print(f"   Creating new vector store...")
            
        # Process document if needed
        if should_process:
            pipeline.process_document(
                args.file_path,
                chunk_size=args.chunk_size, 
                chunk_overlap=args.chunk_overlap,
                persist_directory=db_path
            )
        
        # Process query if provided
        if args.query:
            pipeline.query_document(args.query, top_k=args.top_k)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    main()