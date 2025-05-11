import os
import re
import hashlib
import chromadb
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple

# Import document loaders and text splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from PyPDF2 import PdfReader

from ocr_utils import sanitize_collection_name, compute_hash, extract_clean_metadata

def initialize_embeddings():
    """Initialize HuggingFace Embeddings."""
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("Embedding model loaded.")
    return embeddings

def connect_to_chroma(persist_dir: str):
    """Connect to a Chroma PersistentClient."""
    print(f"Connecting to Chroma DB at {persist_dir}...")
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collections_info = client.list_collections()
        collection_names = [col if isinstance(col, str) else col.get("name") for col in collections_info]
        print(f"Found {len(collection_names)} existing collections")
        return client, collections_info
    except Exception as e:
        print(f"Error connecting to Chroma: {str(e)}")
        return None, []

def load_documents_from_directory(docs_path: str) -> List[Document]:
    """
    Load documents from a directory path.
    
    Args:
        docs_path: Path to directory containing documents
        
    Returns:
        List of Document objects
    """
    docs = []
    for root, _, files in os.walk(docs_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                try:
                    raw_reader = PdfReader(file_path)
                    pdf_metadata = extract_clean_metadata(raw_reader.metadata, file_path)

                    for page_num, page in enumerate(raw_reader.pages):
                        text = page.extract_text()
                        if text and text.strip():  # Avoid empty pages
                            docs.append(Document(
                                page_content=text,
                                metadata={**pdf_metadata, "page": page_num}
                            ))
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
                docs.extend(loader.load())
    return docs

def split_and_prepare_documents(
    docs: List[Document], 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for vector store.
    
    Args:
        docs: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of split document chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Add content hash to each chunk
    for doc in all_splits:
        doc.metadata["content_hash"] = compute_hash(doc.page_content)
        doc.metadata["section"] = "all_sections"
    
    return all_splits

def create_new_vector_store(
    embeddings, 
    persist_dir: str, 
    docs_path: str, 
    collection_name: str
) -> Chroma:
    """
    Create a new Chroma vector store.
    
    Args:
        embeddings: Embedding function
        persist_dir: Directory to persist Chroma store
        docs_path: Path to documents
        collection_name: Name of collection
        
    Returns:
        New Chroma vector store
    """
    print("Loading documents...")
    docs = load_documents_from_directory(docs_path)
    print(f"Loaded {len(docs)} documents.")

    print("Splitting documents into chunks...")
    all_splits = split_and_prepare_documents(docs)
    print(f"Split into {len(all_splits)} chunks.")

    print("Creating new Chroma DB...")
    chroma_store = Chroma.from_documents(
        documents=all_splits,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    print("New Chroma DB created and saved successfully!")
    return chroma_store

def update_vectorstore(
    chroma_store: Chroma,
    new_docs_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> Chroma:
    """
    Update existing Chroma vector store with new documents.
    
    Args:
        chroma_store: Existing Chroma store
        new_docs_path: Path to new documents
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        Updated Chroma vector store
    """
    from ocr_utils import get_existing_hashes
    
    print("Retrieving existing content hashes from Chroma...")
    existing_hashes = get_existing_hashes(chroma_store)
    print(f"Found {len(existing_hashes)} existing hashes.")

    print(f"Loading new documents from: {new_docs_path}")
    raw_docs = load_documents_from_directory(new_docs_path)
    print(f"Loaded {len(raw_docs)} raw documents.")

    if not raw_docs:
        print("No documents found to process.")
        return chroma_store

    print("Splitting documents into chunks...")
    split_docs = split_and_prepare_documents(raw_docs, chunk_size, chunk_overlap)
    print(f"Split into {len(split_docs)} chunks.")

    # Filter out already existing chunks
    unique_chunks = []
    for chunk in split_docs:
        chunk_hash = compute_hash(chunk.page_content)
        chunk.metadata["content_hash"] = chunk_hash
        chunk.metadata["section"] = "all_sections"

        if chunk_hash not in existing_hashes:
            unique_chunks.append(chunk)

    print(f"Found {len(unique_chunks)} new unique chunks to add.")

    if unique_chunks:
        print("Adding new unique chunks to Chroma...")
        chroma_store.add_documents(unique_chunks)
        chroma_store.persist()
        print(f"Successfully added {len(unique_chunks)} new chunks and persisted.")

    return chroma_store

def initialize_or_update_vector_store(
    persist_dir: str, 
    docs_path: str, 
    collection_name: str = None
) -> Chroma:
    """
    Initialize or update a Chroma vector store.
    
    Args:
        persist_dir: Directory to persist Chroma store
        docs_path: Path to documents
        collection_name: Name of collection (optional)
        
    Returns:
        New or updated Chroma vector store
    """
    if collection_name is None:
        collection_name = os.path.basename(docs_path)
        collection_name = sanitize_collection_name(collection_name)
        print(f"Using dynamic collection name: {collection_name}")

    # Initialize embeddings
    embeddings = initialize_embeddings()

    # Connect to Chroma
    client, collections = connect_to_chroma(persist_dir)
    
    # Check if collection exists
    collection_exists = False
    for col in collections:
        col_name = col if isinstance(col, str) else col.get("name")
        if col_name == collection_name:
            collection_exists = True
            break
    
    if collection_exists:
        # Update existing collection
        print(f"Found existing collection: {collection_name}")
        chroma_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        print("Updating Chroma vector store...")
        update_vectorstore(chroma_store, docs_path)
        return chroma_store
    else:
        # Create new collection
        print(f"Creating new collection: {collection_name}")
        chroma_store = create_new_vector_store(embeddings, persist_dir, docs_path, collection_name)
        return chroma_store

def unified_search(
    client,
    persist_dir,
    embeddings,
    query: str,
    k: int = 10,
    filter_section: str = "all_sections"
) -> List[Document]:
    """
    Search across all collections in Chroma.
    
    Args:
        client: Chroma client
        persist_dir: Directory where Chroma is persisted
        embeddings: Embedding function
        query: Search query
        k: Number of results to return
        filter_section: Section to filter by
        
    Returns:
        List of Document objects from search results
    """
    collection_names = client.list_collections()
    print(f"Searching across {len(collection_names)} collections...")

    all_results = []

    for collection in collection_names:
        if isinstance(collection, dict):
            collection_name = collection.get("name")
        else:
            collection_name = collection
        
        print(f"Searching in collection: {collection_name}")

        chroma_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

        try:
            # Use similarity_search_with_score to get the scores
            results = chroma_store.similarity_search_with_score(
                query,
                k=k,
                filter=None  # No filtering for now to maximize results
            )
            all_results.extend(results)
        except Exception as e:
            print(f"Error searching collection {collection_name}: {str(e)}")
            continue

    # Sort all results by similarity score (lower is better)
    all_results.sort(key=lambda x: x[1])

    # Take top k
    top_results = all_results[:k]

    # Extract just the documents
    retrieved_docs = [doc for doc, score in top_results]

    print(f"Found {len(retrieved_docs)} documents after merging.")
    return retrieved_docs