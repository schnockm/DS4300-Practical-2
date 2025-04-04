import os
import json
import time
import redis
import chromadb
import numpy as np
import ollama
import re
import faiss
from redis.commands.search.query import Query
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"
REDIS_HOST = "localhost"
REDIS_PORT = 6379
EMBEDDED_DIR = "/Users/victorzheng/Downloads/DS4300 Lectures/DS4300-Practical-2/data/embedded"
MODEL_ALIASES = {
    "mistral": "mistral:7b",
    "llama": "llama2:7b",
}

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
try:
    redis_client.ping()
    print("✅ Connected to Redis")
except redis.ConnectionError as e:
    print("❌ Redis connection error:", e)
    exit(1)

chroma_client = chromadb.Client()

faiss_index = None
faiss_texts = []

def generate_llm_response(llm_model, prompt):
    start = time.time()
    response = ollama.chat(model=llm_model, messages=[
        {"role": "system", "content": "You are an AI tutor helping a student based on lecture notes."},
        {"role": "user", "content": prompt}
    ])
    elapsed = time.time() - start
    return response["message"]["content"].strip(), elapsed

def load_embedded_data(chunk_size, overlap=0, clean=False):
    texts = []
    embeddings = []

    for file in os.listdir(EMBEDDED_DIR):
        if f"chunks_{chunk_size}_" in file and file.endswith("_embedded.json"):
            with open(os.path.join(EMBEDDED_DIR, file), "r") as f:
                data = json.load(f)
                for entry in data:
                    text = entry.get("chunk")
                    if text is None:
                        raise KeyError("Missing 'chunk' in embedded JSON.")
                    if clean:
                        text = clean_text(text)
                    texts.append(text)
                    embeddings.append(np.array(entry["embedding"], dtype=np.float32))

    texts = apply_overlap(texts, overlap)
    return np.array(embeddings), texts

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[\W_]+", " ", text)
    return " ".join(w for w in text.split() if w not in ENGLISH_STOP_WORDS)

def apply_overlap(chunks, overlap):
    if overlap == 0:
        return chunks
    overlapped_chunks = []
    for i in range(len(chunks)):
        prev = chunks[i - 1][-overlap:] if i > 0 else ""
        overlapped_chunks.append((prev + " " + chunks[i]).strip())
    return overlapped_chunks

def clear_redis():
    redis_client.flushdb()
def create_redis_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass
    redis_client.execute_command(
        f"""FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
            SCHEMA text TEXT embedding VECTOR HNSW 6 TYPE FLOAT32 DIM {VECTOR_DIM} DISTANCE_METRIC {DISTANCE_METRIC}""")

def index_redis(embeddings, texts):
    clear_redis()
    create_redis_index()
    for i, (vec, text) in enumerate(zip(embeddings, texts)):
        redis_client.hset(f"{DOC_PREFIX}{i}", mapping={"text": text, "embedding": vec.tobytes()})

def query_redis(query_vec):
    q = Query("*=>[KNN 10 @embedding $vec AS score]").sort_by("score").return_fields("text", "score").dialect(2)
    results = redis_client.ft(INDEX_NAME).search(q, query_params={"vec": query_vec.tobytes()})
    return [doc.text for doc in results.docs]

def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def index_chroma(embeddings, texts):
    try:
        chroma_client.delete_collection("benchmark")
    except:
        pass
    collection = chroma_client.get_or_create_collection(
        "benchmark",
        metadata={"hnsw:space": "cosine"}
    )
    for i, (vec, text) in enumerate(zip(embeddings, texts)):
        norm_vec = normalize_vec(vec)
        collection.add(documents=[text], embeddings=[norm_vec.tolist()], ids=[str(i)])

def query_chroma(query_vec):
    collection = chroma_client.get_collection("benchmark")
    results = collection.query(
        query_embeddings=[normalize_vec(query_vec).tolist()],
        n_results=10
    )
    return results["documents"][0]

def index_faiss(embeddings, texts):
    global faiss_index, faiss_texts
    norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    faiss_index = faiss.IndexFlatIP(norm_embeddings.shape[1])
    faiss_index.add(norm_embeddings)
    faiss_texts = texts

def query_faiss(query_vec):
    global faiss_index, faiss_texts
    norm_query = normalize_vec(query_vec)
    D, I = faiss_index.search(norm_query.reshape(1, -1), 10)
    return [faiss_texts[i] for i in I[0] if i != -1]

def embed_query(query, model="mistral"):
    if model == "mistral":
        return np.array(ollama.embeddings(model="mistral:7b", prompt=query)["embedding"], dtype=np.float32)[:VECTOR_DIM]
    elif model == "llama":
        return np.array(ollama.embeddings(model="llama2:7b", prompt=query)["embedding"], dtype=np.float32)[:VECTOR_DIM]
    else:
        raise ValueError("Model must be 'mistral' or 'llama'")

def chat():
    chunk_size = int(input("Enter chunk size (100, 500, 1000): "))
    db = input("Choose vector DB (redis, chroma, or faiss): ").lower()
    llm = input("Choose LLM (mistral or llama): ").lower()
    overlap = int(input("Overlap characters (e.g. 0, 50, 100): "))
    clean = input("Clean text? (y/n): ").lower() == "y"
    print("\n🔄 Loading precomputed embeddings...")
    embeddings, texts = load_embedded_data(chunk_size, overlap, clean)

    if db == "redis":
        index_redis(embeddings, texts)
    elif db == "chroma":
        index_chroma(embeddings, texts)
    elif db == "faiss":
        index_faiss(embeddings, texts)
    else:
        print("❌ Invalid vector DB. Choose 'redis', 'chroma', or 'faiss'.")
        return
    print("✅ Indexing complete. Ask your question!\n")

    while True:
        user_query = input("❓ You: ")
        if user_query.lower() in ["exit", "quit"]:
            break
        query_vec = embed_query(user_query, model=llm)

        if db == "redis":
            docs = query_redis(query_vec)
        elif db == "chroma":
            docs = query_chroma(query_vec)
        else:  # faiss
            docs = query_faiss(query_vec)

        print("\n📚 Retrieved context:")
        for i, doc in enumerate(docs):
            print(f"[{i + 1}] Score-based chunk preview:\n{doc[:500]}...\n{'-' * 60}")

        context = "\n".join(docs)
        prompt = f"Answer the question using only the context below.\n\nContext:\n{context}\n\nQuestion: {user_query}"

        response, elapsed = generate_llm_response(llm, prompt)
        print(f"\n🧠 {llm.capitalize()} Response (in {elapsed:.2f} sec):\n{response}\n")

if __name__ == "__main__":
    chat()

