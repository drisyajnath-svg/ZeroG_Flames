import os
from pathlib import Path
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

def process_pdf_to_vectordb(pdf_filepath):
    # Parse and extract raw text across all PDF pages
    extracted_text = ""
    try:
        with open(pdf_filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"Processing {total_pages} pages from {pdf_filepath}...")
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    extracted_text += text + "\n"
                
    except FileNotFoundError:
        print(f"Error: Target file not found at {pdf_filepath}")
        return
        
    print(f"Extracted {len(extracted_text)} characters.")

    # Split text into overlapping chunks to preserve context at boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(extracted_text)
    print(f"Generated {len(chunks)} text chunks.")

    # Load embedding model and connect to persistent Chroma store
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Resolve vector store path dynamically at project root
    base_dir = Path(__file__).resolve().parent.parent
    target_db_path = str(base_dir / "nasa_vectordb")
    chroma_client = chromadb.PersistentClient(path=target_db_path)
    
    collection = chroma_client.get_or_create_collection(
        name="combustion_experiments",
        embedding_function=sentence_transformer_ef
    )

    # Assign deterministic record IDs and batch insert documents
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        ids=ids
    )
    
    print(f"Successfully indexed {len(chunks)} chunks into collection 'combustion_experiments' at {target_db_path}.")

if __name__ == "__main__":
    # Resolve PDF path relative to this script directory
    script_dir = Path(__file__).resolve().parent
    default_path = str(script_dir / "researchers_guide.pdf")
    
    process_pdf_to_vectordb(default_path)
