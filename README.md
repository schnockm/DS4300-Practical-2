# DS4300-Practical-2

Vivek Divakarla, Jordan Walsh, Victor Zheng, Maxwell Schnock

##  Overview
In this project, we are building a Retrieval-Augmented Generation (RAG) system designed to help users query a collection of course notes gathered by the team throughout the semester. The system integrates various technologies and models to provide accurate and contextually relevant answers based on the input query.
Features:
- Document Ingestion: Collects and processes lecture notes in PDF format.
- Indexing: Embeds text chunks into vector databases for efficient search and retrieval.
- Query Handling: Accepts a user query, retrieves relevant context, and generates a response using a locally running LLM (Large Language Model).
- Experimentation: Different configurations are explored, including chunking strategies, embedding models, vector databases, and LLM options.
Tools Used:
- Python for the overall pipeline.
- Ollama for running LLMs locally.
- Vector Databases: Redis Vector DB, Chroma, and FAISS.
- Embedding Models: sentence-transformers/all-MPNet-base-v2 and others.

P## roject Structure

DS4300-Practical-2/
│
├── data/
│   ├── chunked_100/
│   ├── chunked_500/
│   ├── chunked_1000/
│   └── embedded/
├── scripts/
│   ├── chat.py            # Main driver script for querying the system.
│   ├── embed_mpnet.py     # Script to embed chunked text using MPNet.
│   └── preprocess_text.py # Script for preprocessing PDF notes and chunking text.
│   └── miniLM_comparison_script # Script to compare vector DBs and LLMs with miniLM embeddings. 
│   └── miniLM_interactive.py # Interactive chat with the best combination for miniLM. 
└── results/
    └── chat_response_log.json # Log of query responses and metrics.


Setup
Clone the repository:
Copy
git clone https://github.com/schnockm/DS4300-Practical-2.git
Install dependencies: Ensure Python 3.8+ is installed, then install the necessary libraries:

Copy
pip install -r requirements.txt
Install Redis (for Redis Vector DB):
- Follow the Redis installation guide if Redis is not yet installed on your system.
Install Ollama for LLM inference:
- Follow the Ollama installation guide to set up Ollama for local LLM use.

## Running the Project
1. Preprocess the PDF Notes
Run the preprocess_text.py script to extract and chunk the PDF notes into different sizes (100, 500, 1000 tokens).
This will:
- Extract text from all PDFs in the data/ folder.
- Clean the text and chunk it into various sizes (100, 500, 1000 tokens) with optional overlaps.
- Save the chunked text into data/chunked_* directories.
2. Embedding the Text
This will:
- Read the chunked text files from data/chunked_*.
- Generate embeddings using the MPNet model (sentence-transformers/all-mpnet-base-v2).
- Save the embeddings in JSON format inside the data/embedded/ directory.
3. Running the Chat System
This will:
- Prompt you to choose a chunk size (100, 500, or 1000), a vector database (Redis, Chroma, or FAISS), and an LLM (Mistral or Llama).
- Index the precomputed embeddings in the chosen vector database.
- Allow you to input queries, retrieve relevant context, and generate responses using the LLM.
- Log each query, retrieved context, and response to results/chat_response_log.json.

## Experimentation Variables
The system allows for systematic experimentation with different configurations:
- Chunking strategies: Vary the chunk size and overlap (e.g., 0, 50, 100 tokens).
- Embedding Models: Compare the performance of at least 3 embedding models (e.g., all-MPNet-base-v2, InstructorXL, and others).
- Vector Databases: Compare Redis Vector DB, Chroma, and FAISS for indexing and querying.
- LLM Models: Compare different LLM models (e.g., Mistral 7B, Llama 2 7B).
- Text Preprocessing: Experiment with cleaning the text by removing stop words and other noise.

## How to Run MiniLM scripts
- Run preprocess_text.py to generate text in the data folder.
- Run miniLM_comparison_script.py to generate responses based of different vector DBs, chunking strategies, and LLM choices. May need to change location of redis port, and data directories to point to the correct chunked texts.
- Similar to miniLM_comparison_script.py, will need to ensure redis and data directories are updated to run miniLM_interactive.py


