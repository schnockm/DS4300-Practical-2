import os
import time
import redis
import chromadb
import faiss
import numpy as np
import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query

# Redis Setup
redis_client = redis.Redis(host="localhost", port=6380, db=0)

VECTOR_DIM = 384
INDEX_NAME = "doc_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"

def create_redis_index():
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

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="doc_embeddings")

# FAISS Setup
faiss_index = faiss.IndexFlatL2(VECTOR_DIM)
faiss_texts = []

# Loading the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def store_redis_embedding(file_name, text, embedding):
    key = f"{DOC_PREFIX}{file_name}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

def store_chroma_embedding(file_name, text, embedding):
    try:
        chroma_client.delete_collection("benchmark")
    except:
        pass
    collection.add(
        ids=[file_name],
        embeddings=[embedding.tolist()],
        metadatas=[{"text": text}]
    )

def store_faiss_embedding(file_name, text, embedding):
    global faiss_index, faiss_texts
    faiss_index.add(np.array([embedding], dtype=np.float32))
    faiss_texts.append(text)

def process_files(data_dir, db_choice):
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            embedding = model.encode(text)
            if db_choice == "redis":
                store_redis_embedding(filename, text, embedding)
            elif db_choice == "chroma":
                store_chroma_embedding(filename, text, embedding)
            elif db_choice == "faiss":
                store_faiss_embedding(filename, text, embedding)

def query_faiss(query_text, top_k=5):
    #Retrieve the top-K most similar documents from Faiss

    query_embedding = model.encode(query_text).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    retrieved_docs = [faiss_texts[i] for i in indices[0] if i < len(faiss_texts)]
    return retrieved_docs

def query_chroma(query_text, top_k=5):
    #Retrieve the top-K most similar documents from ChromaDB
    
    start_time = time.time()
    query_embedding = model.encode(query_text) 
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )
    end_time = time.time()
    print(f"Query executed in {end_time - start_time:.4f} seconds.")

    retrieved_docs = []
    for i in range(len(results["ids"][0])):
        doc_text = results["metadatas"][0][i]["text"]
        retrieved_docs.append(doc_text)

    return retrieved_docs

def query_redis(query_text, top_k=5):
    #Retrieve the top-K most similar documents from Redis
    query_embedding = model.encode(query_text)

    q = (
        Query("*=>[KNN {} @embedding $vec AS vector_distance]".format(top_k))
        .sort_by("vector_distance")
        .return_fields("text", "vector_distance")
        .dialect(2)
    )
    
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(query_embedding, dtype=np.float32).tobytes()}
    )
    retrieved_docs = []
    for doc in res.docs:  
        retrieved_docs.append(doc)

    return retrieved_docs

def generate_response(query_text, db_choice, llm_choice, metrics):
    start_time = time.time()
    mem_before = psutil.virtual_memory().used
    
    if db_choice == "redis":
        retrieved_docs = query_redis(query_text, top_k=3)
        context = "\n\n".join([doc.text for doc in retrieved_docs])
    elif db_choice == "chroma":
        retrieved_docs = query_chroma(query_text, top_k=3)
        context = "\n\n".join(retrieved_docs)
    elif db_choice == "faiss":
        retrieved_docs = query_faiss(query_text, top_k=3)
        context = "\n\n".join(retrieved_docs)
    else:
        raise ValueError("Invalid db_choice. Choose 'redis', 'chroma', or 'faiss'.")
    
    retrieval_time = time.time() - start_time
    mem_after = psutil.virtual_memory().used
    
    if not retrieved_docs:
        return
    
    #System Prompt
    prompt = f"If the context is not relevant to the query, say 'I don't know'. Use the following information to answer the question:\n\n{context}\n\nQuestion: {query_text}\n\nAnswer:"
    start_time = time.time()
    response = ollama.chat(model=llm_choice, messages=[{"role": "user", "content": prompt}])
    llm_time = time.time() - start_time
    
    #Record relevant statistics
    metrics.append({
        "data_dir": data_dir,
        "db_choice": db_choice,
        "llm_choice": llm_choice,
        "retrieval_time": retrieval_time,
        "memory_usage": (mem_after - mem_before) / (1024 ** 2),
        "llm_time": llm_time,
        "question": query_text,
        "response": response["message"]["content"]
    })

data_dirs = [
    "/Users/vivekdivakarla/DS4300-Practical-2/data/chunked_1000",
    "/Users/vivekdivakarla/DS4300-Practical-2/data/chunked_500",
    "/Users/vivekdivakarla/DS4300-Practical-2/data/chunked_100"
]

db_choices = ["redis", "chroma", "faiss"]
llm_choices = ["llama2:7b", "mistral:7b"]
metrics = []

for data_dir in data_dirs:
    for db_choice in db_choices:
        for llm_choice in llm_choices:
            print(f"Running experiment with Data Directory: {data_dir}, DB: {db_choice}, LLM: {llm_choice}")
            process_files(data_dir, db_choice)
            generate_response("When are linked lists faster than contiguously-allocated lists?", db_choice, llm_choice, metrics)
            generate_response("What is the difference between a list where memory is contiguously allocated and a list where linked structures are used?", db_choice, llm_choice, metrics)
            generate_response("When was Redis originally released?", db_choice, llm_choice, metrics)
            generate_response("Succinctly describe the four components of ACID compliant transactions.", db_choice, llm_choice, metrics)
            generate_response("What is disk-based indexing and why is it important for database systems?", db_choice, llm_choice, metrics)

df = pd.DataFrame(metrics)
df.to_csv("benchmark_results.csv", index=False)
print("Benchmark results saved to benchmark_results.csv")
