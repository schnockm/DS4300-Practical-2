import os
import time
import redis
import numpy as np
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

# Fixed configuration
DATA_DIR = "/Users/vivekdivakarla/DS4300-Practical-2/data/chunked_500"
LLM_CHOICE = "mistral:7b"

# Loading the SentenceTransformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def clear_redis():
    redis_client.flushdb()
    
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

def store_redis_embedding(file_name, text, embedding):
    #Store embeddings in redis
    key = f"{DOC_PREFIX}{file_name}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )

def process_files():
    #Processing Files and Storing in Redis
    
    for filename in sorted(os.listdir(DATA_DIR)):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            embedding = model.encode(text)
            store_redis_embedding(filename, text, embedding)
            

def query_redis(query_text, top_k=3):
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
    
    return res.docs

def generate_response(query_text):
    
    start_time = time.time()
    retrieved_docs = query_redis(query_text, top_k=3)
    retrieval_time = time.time() - start_time
    
    if not retrieved_docs:
        print("No relevant documents found in the database.")
        return
    
    context = "\n\n".join([doc.text for doc in retrieved_docs])
    
    # System prompt
    prompt = f"""If the context is not relevant to the query, say 'I don't know'. 
Use the following information to answer the question:{context} Question: {query_text} Answer:"""
    
    # Generate response from LLM
    start_time = time.time()
    response = ollama.chat(model=LLM_CHOICE, messages=[{"role": "user", "content": prompt}])
    llm_time = time.time() - start_time
    
    # Print stats and response
    print(f"Retrieval time: {retrieval_time:.4f} seconds")
    print(f"LLM response time: {llm_time:.4f} seconds")
    print(f"Total time: {retrieval_time + llm_time:.4f} seconds")
    print(f"\n--- Response ---")
    print(response["message"]["content"])
    
    # Return both the response and the retrieved docs for potential debugging
    return {
        "response": response["message"]["content"],
        "retrieved_contexts": [doc.text for doc in retrieved_docs],
        "stats": {
            "retrieval_time": retrieval_time,
            "llm_time": llm_time
        }
    }

def main():
    print(f"Data source: {DATA_DIR}")
    print(f"LLM: {LLM_CHOICE}")
    print(f"Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    
    # Check if Redis is running and properly connected
    try:
        redis_client.ping()
        print("Redis connection successful!")
    except redis.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Redis. Make sure Redis is running on localhost:6380")
        return
    
    # Create Redis index and process files
    clear_redis()
    create_redis_index()
    process_files()
    
    print("Type your questions below or 'exit' to quit")
    
    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
            
        generate_response(query)

if __name__ == "__main__":
    main()