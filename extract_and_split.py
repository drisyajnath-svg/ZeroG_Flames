from pathlib import Path
from PyPDF2 import PdfReader
import chromadb
import chromadb

# In newer LangChain versions, text splitters are in langchain_text_splitters
# In newer LangChain versions, text splitters are in langchain_text_splitters
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extracts all text from a PDF file into a single string using PyPDF2."""
    reader = PdfReader(str(pdf_path))
    full_text = []

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text.append(page_text)

    return "\n".join(full_text)

def main():
    # Replace with the path to your target PDF file
    pdf_file_path = r"C:/Users/Adithyakrishna U/Desktop/NASA RESOURCES/ICES 2021-266 0503-2021 final.pdf"

    if not Path(pdf_file_path).exists():
        print(f"File '{pdf_file_path}' not found. Please provide a valid PDF path.")
        return

    # 1. Extract all text into a single string
    print(f"Extracting text from: {pdf_file_path} ...")
    giant_text = extract_text_from_pdf(pdf_file_path)
    print(f"Total characters extracted: {len(giant_text)}")

    # 2. Initialize RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Pass the giant text string into the splitter
    chunks = text_splitter.split_text(giant_text)
    print(f"Total chunks created: {len(chunks)}\n")

    # ==========================================
    # NEW CODE: SAVE TO VECTOR DATABASE
    # ==========================================
    print("Connecting to Vector Database...")
    chroma_client = chromadb.PersistentClient(path="./nasa_vectordb")
    
    # CRITICAL: Make sure "nasa_psi_data" is the exact same collection name your backend uses!
    collection = chroma_client.get_or_create_collection(name="nasa_psi_data") 

    # Extract just the filename (e.g., "ICES 2021-266 0503-2021 final") to use as a unique ID
    pdf_name = Path(pdf_file_path).stem 
    
    metadatas = [{"source": pdf_name, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]

    # Add the chunks to the database
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"SUCCESS: Added {len(chunks)} chunks from '{pdf_name}' to the database!")
    # ==========================================
    # NEW CODE: SAVE TO VECTOR DATABASE
    # ==========================================
    print("Connecting to Vector Database...")
    chroma_client = chromadb.PersistentClient(path="./nasa_vectordb")
    
    # CRITICAL: Make sure "nasa_psi_data" is the exact same collection name your backend uses!
    collection = chroma_client.get_or_create_collection(name="nasa_psi_data") 

    # Extract just the filename (e.g., "ICES 2021-266 0503-2021 final") to use as a unique ID
    pdf_name = Path(pdf_file_path).stem 
    
    metadatas = [{"source": pdf_name, "chunk_index": i} for i in range(len(chunks))]
    ids = [f"{pdf_name}_chunk_{i}" for i in range(len(chunks))]

    # Add the chunks to the database
    collection.add(
        documents=chunks,
        metadatas=metadatas,
        ids=ids
    )
    
    print(f"SUCCESS: Added {len(chunks)} chunks from '{pdf_name}' to the database!")

if __name__ == "__main__":
    main()