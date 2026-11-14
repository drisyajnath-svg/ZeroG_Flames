from pathlib import Path
from PyPDF2 import PdfReader

# In newer LangChain versions, text splitters are in langchain_text_splitters,
# while in classic LangChain they are imported from langchain.text_splitter.
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

    # Combine all pages into a single giant string
    return "\n".join(full_text)


def main():
    # Replace with the path to your target PDF file
    pdf_file_path = "combustion_science.pdf"

    if not Path(pdf_file_path).exists():
        print(f"File '{pdf_file_path}' not found. Please provide a valid PDF path.")
        return

    # 1. Extract all text into a single string
    print(f"Extracting text from: {pdf_file_path} ...")
    giant_text = extract_text_from_pdf(pdf_file_path)
    print(f"Total characters extracted: {len(giant_text)}")

    # 2. Initialize RecursiveCharacterTextSplitter
    # chunk size = 1000 characters, chunk overlap = 200 characters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # 3. Pass the giant text string into the splitter
    chunks = text_splitter.split_text(giant_text)
    print(f"Total chunks created: {len(chunks)}\n")

    # Display preview of the first few chunks
    for idx, chunk in enumerate(chunks[:3], start=1):
        print(f"--- Chunk {idx} ({len(chunk)} chars) ---")
        print(chunk[:300] + ("..." if len(chunk) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
