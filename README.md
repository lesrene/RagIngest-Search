# Overview

This pipeline processes a directory of PDFs, extracts text, embeds the text using a chosen model, and stores the embeddings in a vector database for efficient retrieval.

## Prerequisites

Ensure you have the depndencies in the requirements.txt file installed

## Running the Pipeline

To execute the pipeline with the default configuration, run: python pipeline.py

## Modifying Pipeline Variables

You can override default variables dynamically using the -m flag. Examples:

python pipeline.py -m embedding_model=all_Mini
python pipeline.py -m vector_db=faiss
python pipeline.py -m chunk_size=500 chunk_overlap=50

## Troubleshooting

- Sometimes you need to make sure that the connection to the docker container that holds the vector database is actually working. If you're using an redis or chroma instance, make sure it's running in DOcker before executing your script.  
- Sometimes you may need to delete/remove the current index built and rebuild when running a new vector database. 

For any issues, please refer to the error messages and logs, or reach out for support!

## Logging and Performance Tracking

The pipeline logs processing times and memory usage in performance_log.csv. You can analyze different configurations by comparing the logged values.


# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 
