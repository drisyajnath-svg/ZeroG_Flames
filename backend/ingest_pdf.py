import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

def process_pdf_to_vectordb(pdf_filepath):
    print(f"--- Step 1: Opening {pdf_filepath} ---")
    
    # 1. Extract Text from PDF
    extracted_text = ""
    try:
        with open(pdf_filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            print(f"Found {total_pages} pages. Extracting text...")
            
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                # Extract text and add a newline to keep paragraphs separated
                extracted_text += page.extract_text() + "\n"
                
    except FileNotFoundError:
        print(f"ERROR: Could not find {pdf_filepath}. Check the file name and location!")
        return
        
    print(f"Extraction complete! Total characters read: {len(extracted_text)}")

    # 2. Chunk the Text
    print("\n--- Step 2: Chunking the text into smaller pieces ---")
    # We use 1000 characters per chunk, with a 200 character overlap so sentences don't get cut in half.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(extracted_text)
    print(f"Created {len(chunks)} text chunks.")

    # 3. Setup Vector Database (ChromaDB)
    print("\n--- Step 3: Loading embedding model and creating Vector DB ---")
    # This downloads a small, fast AI model to your computer to convert text to numbers.
    # Your RTX 4050 will handle this easily.
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create a local database folder named "nasa_vectordb" in your current directory
    chroma_client = chromadb.PersistentClient(path="./nasa_vectordb")
    
    # Create a collection (like a table in a database) for your combustion data
    collection = chroma_client.get_or_create_collection(
        name="combustion_experiments",
        embedding_function=sentence_transformer_ef
    )

    # 4. Add chunks to the database
    print("\n--- Step 4: Adding chunks to the database (This might take a minute) ---")
    
    # We need to give each chunk a unique ID (e.g., "chunk_0", "chunk_1")
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    
    # Add the documents and their IDs to ChromaDB
    collection.add(
        documents=chunks,
        ids=ids
    )
    
    print("\n✅ SUCCESS! The PDF is now embedded in the database.")
    print("A new folder named 'nasa_vectordb' has been created in your directory.")

# --- Run the script ---
if __name__ == "__main__":
    # IMPORTANT: Change this string to the exact name of the PDF you downloaded!
    pdf_filename = r"D:/NASA/ZeroG_Flames/backend/researchers_guide.pdf"
    
    process_pdf_to_vectordb(pdf_filename)
