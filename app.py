import streamlit as st
import os
from pathlib import Path
import time
import tempfile

# Import our custom modules
from ocr_utils import extract_text_from_pdf_ocr, sanitize_collection_name
from vector_store import initialize_embeddings, connect_to_chroma, initialize_or_update_vector_store
from rag_pipeline import setup_rag_pipeline, ask_question

# Page configuration
st.set_page_config(
    page_title="Document Intelligence App",
    page_icon="📑",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 500;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);
    }
    .success-text {
        color: #28a745;
        font-weight: 500;
    }
    .warning-text {
        color: #ffc107;
        font-weight: 500;
    }
    .info-text {
        color: #17a2b8;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'persist_dir' not in st.session_state:
    st.session_state.persist_dir = tempfile.mkdtemp()
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'client' not in st.session_state:
    st.session_state.client = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'has_uploaded' not in st.session_state:
    st.session_state.has_uploaded = False
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = "default_collection"
if 'api_keys_set' not in st.session_state:
    st.session_state.api_keys_set = False

# Main header
st.markdown('<p class="main-header">Document Intelligence Center</p>', unsafe_allow_html=True)
st.markdown("Unlock insights from your documents with OCR and AI-powered search.")

# Sidebar for settings and configuration
with st.sidebar:
    st.header("Settings")
    
    # API Keys section
    st.subheader("API Keys")
    with st.expander("Configure API Keys", expanded=not st.session_state.api_keys_set):
        mistral_key = st.text_input("Mistral API Key", type="password", 
                                     value=os.environ.get("MISTRAL_API_KEY", ""))
        langsmith_key = st.text_input("LangSmith API Key", type="password", 
                                       value=os.environ.get("LANGSMITH_API_KEY", ""))
        
        if st.button("Save API Keys"):
            os.environ["MISTRAL_API_KEY"] = mistral_key
            os.environ["LANGSMITH_API_KEY"] = langsmith_key
            os.environ["LANGSMITH_TRACING"] = "true"
            st.session_state.api_keys_set = True
            st.success("API keys saved successfully!")
    
    # Vector Store Settings
    st.subheader("Vector Store Settings")
    collection_name_input = st.text_input(
        "Collection Name", 
        value=st.session_state.collection_name,
        help="Name for the document collection in the vector database"
    )
    if collection_name_input != st.session_state.collection_name:
        st.session_state.collection_name = sanitize_collection_name(collection_name_input)
        st.info(f"Sanitized collection name: {st.session_state.collection_name}")
    
    # Initialize embeddings and chromadb client
    if st.button("Initialize System"):
        with st.spinner("Initializing embeddings and database connection..."):
            st.session_state.embeddings = initialize_embeddings()
            st.session_state.client, collections = connect_to_chroma(st.session_state.persist_dir)
            
            if collections:
                st.write(f"Found {len(collections)} existing collections:")
                for col in collections:
                    col_name = col if isinstance(col, str) else col.get("name")
                    st.write(f"- {col_name}")
            
            if st.session_state.api_keys_set:
                st.session_state.rag_pipeline = setup_rag_pipeline(
                    st.session_state.client,
                    st.session_state.persist_dir, 
                    st.session_state.embeddings
                )
                st.success("System initialized successfully!")
            else:
                st.warning("Please set API keys before initializing the RAG pipeline.")

# Main app tabs
tab1, tab2, tab3 = st.tabs(["Document Upload", "Document Search", "Ask Questions"])

# Document Upload Tab
with tab1:
    st.markdown('<p class="sub-header">Upload Documents</p>', unsafe_allow_html=True)
    st.write("Upload PDF documents to process with OCR and add to the vector store.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Upload one or more PDF files to process"
    )
    
    use_ocr = st.checkbox("Apply OCR to documents", value=True, 
                          help="Use Optical Character Recognition to extract text from images in PDFs")
    
    if st.button("Process Documents") and uploaded_files:
        if not st.session_state.embeddings:
            st.error("Please initialize the system first!")
        else:
            # Create temp directory for files
            temp_dir = Path(tempfile.mkdtemp())
            st.session_state.docs_path = str(temp_dir)
            
            # Save uploaded files to temp directory
            progress_bar = st.progress(0)
            file_status = st.empty()
            
            for i, uploaded_file in enumerate(uploaded_files):
                progress = (i + 1) / len(uploaded_files)
                file_path = temp_dir / uploaded_file.name
                
                file_status.info(f"Processing: {uploaded_file.name}")
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Apply OCR if enabled
                if use_ocr:
                    with st.spinner(f"Applying OCR to {uploaded_file.name}..."):
                        extracted_text = extract_text_from_pdf_ocr(str(file_path))
                        st.write(f"Extracted {len(extracted_text.split())} words from {uploaded_file.name}")
                
                progress_bar.progress(progress)
            
            # Initialize or update vector store with documents
            with st.spinner("Adding documents to vector store..."):
                try:
                    initialize_or_update_vector_store(
                        st.session_state.persist_dir,
                        st.session_state.docs_path,
                        st.session_state.collection_name
                    )
                    st.session_state.has_uploaded = True
                    st.success("Documents processed and added to vector store successfully!")
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")

# Document Search Tab
with tab2:
    st.markdown('<p class="sub-header">Search Documents</p>', unsafe_allow_html=True)
    st.write("Search through your processed documents using semantic search.")
    
    search_query = st.text_input("Enter your search query", placeholder="Search for...")
    topk = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
    if st.button("Search") and search_query:
        if not st.session_state.has_uploaded or not st.session_state.embeddings:
            st.warning("Please upload and process documents first!")
        else:
            with st.spinner("Searching documents..."):
                try:
                    from vector_store import unified_search
                    results = unified_search(
                        st.session_state.client,
                        st.session_state.persist_dir,
                        st.session_state.embeddings,
                        search_query,
                        k=topk
                    )
                    
                    if results:
                        st.success(f"Found {len(results)} relevant documents")
                        for i, doc in enumerate(results):
                            with st.expander(f"Result {i+1}: {doc.metadata.get('file_name', 'Document')}"):
                                st.markdown(f"""
                                <div class="card">
                                    <strong>Source:</strong> {doc.metadata.get('source', 'Unknown')}
                                    <strong>Page:</strong> {doc.metadata.get('page', 'N/A')}
                                    <hr/>
                                    {doc.page_content[:500]}... <em>(truncated)</em>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No results found for your query.")
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

# Ask Questions Tab
with tab3:
    st.markdown('<p class="sub-header">Ask Questions</p>', unsafe_allow_html=True)
    st.write("Ask questions about your documents using the RAG pipeline.")
    
    question = st.text_input("Ask a question about your documents", placeholder="What information can I find about...?")
    
    if st.button("Submit Question") and question:
        if not st.session_state.rag_pipeline or not st.session_state.has_uploaded:
            st.warning("Please initialize the system and upload documents first!")
        elif not st.session_state.api_keys_set:
            st.error("Please set API keys in the sidebar first!")
        else:
            with st.spinner("Processing your question..."):
                try:
                    start_time = time.time()
                    answer = ask_question(question)
                    end_time = time.time()
                    
                    st.markdown(f"""
                    <div class="card">
                        <strong>Question:</strong> {question}
                        <hr/>
                        <strong>Answer:</strong><br/>
                        {answer}
                        <br/><br/>
                        <em class="info-text">Response time: {end_time - start_time:.2f} seconds</em>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """<div style="text-align: center; color: #666;">
    Document Intelligence App | Created with Streamlit
    </div>""", 
    unsafe_allow_html=True
)