import os
import ollama
import redis
import numpy as np
from redis.commands.search.query import Query

# Define paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKED_DIR = os.path.join(PROJECT_DIR, "data", "chunked_500")

# Redis config
redis_client = redis.Redis(host="localhost", port=6380, db=0)
VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

# Create Redis index
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# Generate embedding
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Store in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )
    print(f"Stored embedding for chunk {doc_id}")

# Index all PDF chunks in txt form
def index_pdf_chunks():
    create_hnsw_index()

    chunk_files = [f for f in os.listdir(CHUNKED_DIR) if f.endswith(".txt")]
    doc_id = 0

    for filename in chunk_files:
        with open(os.path.join(CHUNKED_DIR, filename), "r", encoding="utf-8") as f:
            content = f.read().split("\n\n---\n\n")

        for chunk in content:
            if chunk.strip():
                embedding = get_embedding(chunk)
                store_embedding(str(doc_id), chunk, embedding)
                doc_id += 1

    print("All PDF chunks indexed.")

# Run RAG

def rag_answer(question: str, k: int = 3, model: str = "llama3.2:latest"):
    print(f"\nQuestion: {question}")
    embedding = get_embedding(question)

    q = (
        Query("*=>[KNN 3 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )

    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    retrieved_chunks = []
    for doc in res.docs:
        raw_text = redis_client.hget(doc.id, "text")
        if raw_text:
            chunk_text = raw_text.decode("utf-8")
            retrieved_chunks.append(chunk_text)


    context = "\n".join(retrieved_chunks)

    rag_prompt = f"""You are a helpful AI assistant. Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    response = ollama.chat(model=model, messages=[{"role": "user", "content": rag_prompt}])
    print("\nAnswer:", response["message"]["content"])

if __name__ == "__main__":
    index_pdf_chunks()
    # Example query
    rag_answer("What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?")
