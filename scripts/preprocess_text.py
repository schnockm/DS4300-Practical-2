import os
import re
import nltk
import pdfplumber
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Get the root project directory
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input folder with raw PDFs
RAW_NOTES_DIR = os.path.join(PROJECT_DIR, "data", "raw_data")

# Chunk sizes
CHUNK_SIZES = [100, 500, 1000]

# Output directories for each chunk size
CHUNKED_DIRS = {
    size: os.path.join(PROJECT_DIR, "data", f"chunked_{size}")
    for size in CHUNK_SIZES
}
for dir_path in CHUNKED_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

def clean_text(text):
    """Clean and normalize text extracted from PDFs."""
    text = text.lower()
    text = re.sub(r'[•▪︎●○■▪▶►]', '-', text)  # Fix bullet points
    text = re.sub(r'[^\w\s\-\n]', '', text)    # Remove most special chaacters
    text = re.sub(r'\n{2,}', '\n\n', text)     # Keep paragraph breaks
    text = re.sub(r'\s+', ' ', text)           # Remove spaces
    return text.strip()

def extract_text_from_pdf(pdf_path):
    """Extract and clean text from a single PDF."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return clean_text(text)

def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping chunks."""
    tokens = word_tokenize(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def process_all_pdfs():
    """Extract, clean, chunk, and save all PDFs."""

    pdf_files = [f for f in os.listdir(RAW_NOTES_DIR) if f.endswith(".pdf")]

    for filename in pdf_files:
        pdf_path = os.path.join(RAW_NOTES_DIR, filename)
        print(f"Processing: {filename}")
        
        # Call previous function to extract and clean text
        text = extract_text_from_pdf(pdf_path)

        if not text:
            print(f"No text found in {filename}")
            continue

        for size in CHUNK_SIZES:
            overlap = max(10, size // 10)
            chunks = chunk_text(text, chunk_size=size, overlap=overlap)
            output_filename = filename.replace(".pdf", f"_chunks_{size}.txt")
            output_path = os.path.join(CHUNKED_DIRS[size], output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                for idx, chunk in enumerate(chunks):
                    f.write(f"Chunk {idx + 1}:\n{chunk}\n\n---\n\n")

            print(f"Saved {len(chunks)} chunks of size {size} to: {output_path}")

if __name__ == "__main__":
    process_all_pdfs()
    print("\nAll PDFs processed and chunked.")
