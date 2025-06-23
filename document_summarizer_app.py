import streamlit as st
import os
import time
from document_summarizer import RAGPipeline, SmartDocumentProcessor

# Set page configuration
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    .main {padding-top: 1rem;}
    .stApp {max-width: 1200px; margin: 0 auto;}
    .result-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 1.5rem;
        background-color: #f8f9fa;
        color: #000 !important;  /* Fixed: was text-color */
    }
    .chunk-box {
        border: 1px solid #e6f3ff;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        background-color: #f0f8ff;
        color: #000 !important;  /* Fixed: was text-color */
    }
    .score-badge {
        background-color: #4CAF50;
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.8rem;
    }
    /* Additional styling to ensure text is visible */
    .result-box p, .chunk-box p {
        color: #000 !important;
    }
    /* Make markdown text in Streamlit elements visible */
    .stMarkdown {
        color: #000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for persistent data
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'db_path' not in st.session_state:
    st.session_state.db_path = None

# Title and description
st.title("📄 Document Summarization & QA System")
st.markdown("Upload a document and ask questions about its content using RAG technology.")

# Sidebar for file upload and parameters
with st.sidebar:
    st.header("📁 Document Upload")
    
    uploaded_file = st.file_uploader("Choose a document", type=["pdf", "txt", "md"])
    
    if uploaded_file:
        # Save the uploaded file to a temporary location
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File uploaded: {uploaded_file.name}")
        st.session_state.uploaded_file_name = uploaded_file.name

    st.header("⚙️ Parameters")
    
    model_options = [
        "llama3-70b-8192",
    ]

    model = st.selectbox("LLM Model", model_options, index=0)

    embedding_options = [
        "all-MiniLM-L6-v2",
    ]
    
    embedding_model = st.selectbox("Embedding Model", embedding_options, index=0)
    
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=4000, value=2000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50)
    top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=6)
    
    st.markdown("---")
    force_reprocess = st.checkbox("Force Reprocessing", help="Process document again even if it was processed before")
    
    if uploaded_file:
        if st.button("Process Document"):
            # Create vector DB directory
            vector_db_dir = "vector_db"
            os.makedirs(vector_db_dir, exist_ok=True)
            
            # Create file-specific database path
            file_basename = os.path.basename(file_path)
            db_path = os.path.join(vector_db_dir, file_basename.replace('.', '_'))
            st.session_state.db_path = db_path
            
            # Show processing status
            with st.spinner("Initializing Pipeline..."):
                try:
                    # Initialize the pipeline
                    pipeline = RAGPipeline(
                        embedding_model=embedding_model,
                        llm_model=model
                    )
                    st.session_state.pipeline = pipeline
                    
                    # Check if we need to process the document
                    should_process = force_reprocess
                    if not should_process:
                        if not (os.path.exists(f"{db_path}.index") and os.path.exists(f"{db_path}.chunks")):
                            should_process = True
                    
                    if should_process:
                        with st.spinner(f"Processing document: {file_basename}"):
                            start_time = time.time()
                            pipeline.process_document(
                                file_path,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                persist_directory=db_path
                            )
                            processing_time = time.time() - start_time
                            st.success(f"✅ Document processed in {processing_time:.2f}s")
                    else:
                        # Load existing database
                        with st.spinner("Loading existing vector database..."):
                            pipeline.vector_store.load_database(db_path)
                            st.success(f"✅ Loaded existing vector database for {file_basename}")
                    
                    st.session_state.processing_complete = True
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Main content area
if st.session_state.processing_complete and st.session_state.pipeline:
    st.header("🔍 Document Query")
    
    query = st.text_input("Ask a question about the document:")
    
    if query:
        with st.spinner("Generating answer..."):
            # Record the start time
            start_time = time.time()
            
            # Embed query
            query_embedding = st.session_state.pipeline.embedding_manager.embed_chunks([query])
            
            # Retrieve relevant chunks
            results = st.session_state.pipeline.vector_store.retrieve_relevant_chunks(
                query_embedding, 
                top_k=top_k
            )
            
            # Generate response
            response = st.session_state.pipeline.generator.generate_response(
                query, 
                results, 
                top_k=top_k
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
        
        # Display results
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("### 📝 Generated Answer")
            st.markdown(f"<div class='result-box'>{response}</div>", unsafe_allow_html=True)
            
            st.markdown(f"⏱️ Processing time: {processing_time:.2f} seconds")
        
        with col2:
            st.markdown("### 🔢 Retrieved Chunks")
            
            for i, (chunk, score) in enumerate(results):
                preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
                st.markdown(
                    f"""<div class='chunk-box'>
                        <span class='score-badge'>{score:.3f}</span>
                        <strong>Chunk {i+1}</strong><br>
                        {preview}
                    </div>""", 
                    unsafe_allow_html=True
                )
else:
    # Show initial instructions
    st.info("👈 Please upload a document and set parameters in the sidebar to get started.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, SentenceTransformers, and Groq API")