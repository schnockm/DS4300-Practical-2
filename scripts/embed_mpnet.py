import os
import json
import time
import psutil
from sentence_transformers import SentenceTransformer

# Load the embedding model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Base chunked data directories
CHUNKED_DIRS = [
    "/Users/victorzheng/Downloads/DS4300 Lectures/DS4300-Practical-2/data/chunked_100",
    "/Users/victorzheng/Downloads/DS4300 Lectures/DS4300-Practical-2/data/chunked_500",
    "/Users/victorzheng/Downloads/DS4300 Lectures/DS4300-Practical-2/data/chunked_1000"
]

# Output directory for saving embedded JSON files
OUTPUT_DIR = "/Users/victorzheng/Downloads/DS4300 Lectures/DS4300-Practical-2/data/embedded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def measure_memory():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # Memory in MB

def embed_text():
    for chunk_dir in CHUNKED_DIRS:
        for filename in os.listdir(chunk_dir):
            if not filename.endswith(".txt"):
                continue

            input_path = os.path.join(chunk_dir, filename)
            print(f"Embedding chunks from: {input_path}")

            with open(input_path, "r", encoding="utf-8") as f:
                content = f.read()

            chunks = [chunk.strip() for chunk in content.split("---") if chunk.strip()]
            texts = []
            for chunk in chunks:
                if chunk.startswith("Chunk"):
                    parts = chunk.split("\n", 1)
                    if len(parts) == 2:
                        texts.append(parts[1].strip())

            start_time = time.time()
            memory_before = measure_memory()

            embeddings = [model.encode(chunk).tolist() for chunk in texts]

            end_time = time.time()
            memory_after = measure_memory()

            output_filename = filename.replace(".txt", "_embedded.json")
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([{"chunk": chunk, "embedding": emb} for chunk, emb in zip(texts, embeddings)], f, indent=4)

            print(f"Saved {len(texts)} embeddings to: {output_path}")
            print(f"Time: {end_time - start_time:.2f}s | Memory: {memory_after - memory_before:.2f} MB")

if __name__ == "__main__":
    embed_text()
    print("\nAll chunk files embedded.")