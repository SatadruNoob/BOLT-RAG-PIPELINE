import os
import re
import hashlib
import pytesseract
from pdf2image import convert_from_path
from langchain_core.documents import Document

def sanitize_collection_name(name: str) -> str:
    """Sanitize a string to be used as a collection name in Chroma."""
    name = name.replace(" ", "_")
    name = re.sub(r'[^a-zA-Z0-9_\-]', '', name)
    return name[:63]

def compute_hash(content: str) -> str:
    """Compute SHA-256 hash of the document content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def extract_text_from_pdf_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF using OCR.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"OCR failed for {pdf_path}: {str(e)}")
        return ""

def extract_clean_metadata(raw_metadata, file_path):
    """
    Extract clean metadata from PDF metadata.
    
    Args:
        raw_metadata: Raw PDF metadata
        file_path: Path to the PDF file
        
    Returns:
        Cleaned metadata dictionary
    """
    cleaned = {
        "source": file_path,
        "file_name": os.path.basename(file_path)
    }
    if raw_metadata:
        for key, value in raw_metadata.items():
            # Only include serializable and clean fields
            if isinstance(key, str) and isinstance(value, (str, int, float)):
                cleaned[key] = str(value)
    return cleaned

def get_existing_hashes(chroma_store) -> set:
    """
    Get existing content hashes from Chroma store.
    
    Args:
        chroma_store: Chroma vector store instance
        
    Returns:
        Set of existing content hashes
    """
    existing_hashes = set()
    try:
        results = chroma_store._collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        for metadata in metadatas:
            if metadata and "content_hash" in metadata:
                existing_hashes.add(metadata["content_hash"])
    except Exception as e:
        print(f"Error retrieving existing hashes from Chroma: {str(e)}")
    return existing_hashes

def ocr_and_update_chroma(doc_dir, persist_dir, chroma_store=None):
    """
    Apply OCR to PDF documents and update Chroma store.
    
    Args:
        doc_dir: Directory containing PDF documents
        persist_dir: Directory to persist Chroma store
        chroma_store: Existing Chroma store instance (optional)
        
    Returns:
        Updated or new Chroma store instance
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    import chromadb
    
    collection_name = sanitize_collection_name(os.path.basename(doc_dir))
    print(f"Using collection name: `{collection_name}`")
    
    # Initialize embeddings if not provided
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Initialize Chroma store if not provided
    if not chroma_store:
        chroma_store = Chroma(
            client=chromadb.PersistentClient(path=persist_dir),
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
    
    existing_hashes = get_existing_hashes(chroma_store)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    new_chunks = []
    
    print("Scanning and applying OCR to all PDFs...")
    
    for i, file in enumerate(sorted(os.listdir(doc_dir))):
        path = os.path.join(doc_dir, file)
        if file.lower().endswith(".pdf"):
            print(f"Processing: {file}")
            text = extract_text_from_pdf_ocr(path)
            
            if not text.strip():
                print(f"OCR returned empty text for {file}")
                continue
            
            chunks = text_splitter.split_documents([Document(page_content=text, metadata={})])
            filtered_chunks = []
            
            for chunk in chunks:
                chunk_hash = compute_hash(chunk.page_content)
                if chunk_hash not in existing_hashes:
                    chunk.metadata = {
                        "source": path,
                        "file_name": file,
                        "content_hash": chunk_hash,
                        "section": "ocr_recovered"
                    }
                    filtered_chunks.append(chunk)
            
            if filtered_chunks:
                print(f"Found {len(filtered_chunks)} new chunks to add from {file}")
                new_chunks.extend(filtered_chunks)
            else:
                print(f"All OCR chunks already exist in Chroma for {file}")
    
    if new_chunks:
        print(f"Adding {len(new_chunks)} new OCR-recovered chunks...")
        chroma_store.add_documents(new_chunks)
        chroma_store.persist()
        print("Chroma DB updated successfully!")
    else:
        print("No new OCR chunks to add.")
    
    return chroma_store