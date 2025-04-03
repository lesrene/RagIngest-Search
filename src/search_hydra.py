import redis
import json
import numpy as np
import hydra
import faiss
import chromadb
from omegaconf import DictConfig
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
import time
import resource
import csv

def track_performance(func, *args, **kwargs):
    start_time = time.time()
    memory_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # in KB (Linux/Mac)

    # Execute the function
    result = func(*args, **kwargs)

    memory_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    end_time = time.time()

    # Calculate time and memory usage
    elapsed_time = end_time - start_time
    memory_used = memory_after - memory_before

    return result, elapsed_time, memory_used

def get_embedding(text: str, cfg, model=None) -> list:
    if cfg.embedding_model.name == "nomic-embed-text":
        response = ollama.embeddings(model=cfg.embedding_model.name, prompt=text)
        return response["embedding"]
    elif model:
        return model.encode(text).tolist()
    else:
        raise ValueError("No valid embedding model provided.")

# Function to search in Redis
def search_redis(query, cfg, top_k=3):
    redis_client = redis.StrictRedis(host=cfg.vector_db.host, port=cfg.vector_db.port, decode_responses=True)

    query_embedding = get_embedding(query, cfg)
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        results = redis_client.ft(cfg.vector_db.index_name).search(
            q, query_params={"vec": query_vector}
        )

        return [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

    except Exception as e:
        print(f"Redis search error: {e}")
        return []

# Function to search in FAISS
def search_faiss(query, cfg, top_k=3):
    index = faiss.read_index(cfg.vector_db.index_path)
    
    query_embedding = np.array(get_embedding(query, cfg), dtype=np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Reshape for FAISS

    distances, indices = index.search(query_embedding, top_k)

    return [{"index": int(idx), "similarity": float(dist)} for idx, dist in zip(indices[0], distances[0]) if idx != -1]

# Function to search in ChromaDB
def search_chroma(query, cfg, top_k=3):
    client = chromadb.PersistentClient(path=cfg.vector_db.db_path)
    collection = client.get_or_create_collection(cfg.vector_db.collection_name)

    query_embedding = get_embedding(query, cfg)
    
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    
    return [
        {"file": res["file"], "page": res["page"], "chunk": res["chunk"], "similarity": res["similarity"]}
        for res in results["documents"][0]
    ]


# Function to select the correct search method
def search_embeddings(query, cfg, top_k=3):
    if cfg.vector_db.name == "redis":
        return search_redis(query, cfg, top_k)
    elif cfg.vector_db.name == "faiss":
        return search_faiss(query, cfg, top_k)
    elif cfg.vector_db.name == "chroma":
        return search_chroma(query, cfg, top_k)
    else:
        raise ValueError(f"Unsupported vector database: {cfg.vector_db.name}")

# Function to generate response with formatted output
def generate_rag_response(query, context_results, cfg):
    if not context_results:
        return "I couldn't find relevant context for this query."

    # Format context with bullet points
    context_str = "\n".join(
        [
            f"- **File:** {result.get('file', 'Unknown file')}\n"
            f"  - **Page:** {result.get('page', 'Unknown page')}\n"
            f"  - **Chunk:** {result.get('chunk', 'Unknown chunk')}\n"
            f"  - **Similarity:** {float(result.get('similarity', 0)):.2f}\n"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
Use the following context to answer the query as accurately as possible. 
If the context is not relevant, say 'I don't know'.

**Context:**  
{context_str}

**Query:** {query}  

**Answer:**"""

    # Generate response
    response = ollama.chat(
        model=cfg.llm.name, messages=[{"role": "user", "content": prompt}]
    )
    raw_response = response["message"]["content"]

    # 🔹 Format response with bullet points (if applicable)
    formatted_response = "\n".join([f"- {line.strip()}" for line in raw_response.split("\n") if line.strip()])

    return formatted_response


# Function to process a file with 5 prompts
def process_prompt_file(file_path, cfg):
    try:
        # Read the file and get the prompts
        with open(file_path, "r", encoding="utf-8") as file:
            prompts = [line.strip() for line in file.readlines() if line.strip()]

        if len(prompts) < 1:
            print(" Error: The input file must contain at least 1 prompt.")
            return

        responses = []

        # Process each prompt
        for prompt in prompts:
            context_results = search_embeddings(prompt, cfg)
            response = generate_rag_response(prompt, context_results, cfg)
            responses.append(f"Query: {prompt}\nResponse: {response}\n")

        for response in responses:
            print(response)

    except Exception as e:
        print(f" Error processing file: {e}")

@hydra.main(config_path="/Users/lesrene/Desktop/DS4300/RagIngestAndSearch/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Chunk Size: {cfg.chunk_size}")
    print(f"Chunk Overlap: {cfg.chunk_overlap}")
    print(f"Vector Database: {cfg.vector_db.name}")  
    print(f"Embedding Model: {cfg.embedding_model.name}")
    print(f"LLM: {cfg.llm.name}\n")
    #file_path = "prompts.txt"  
    file_path = cfg.prompts_dir.directory
    #process_prompt_file(file_path, cfg)

    result2, elapsed_time2, memory_used2 = track_performance(process_prompt_file, file_path, cfg)

    with open("performance_log.csv", mode='r+', newline='') as file:
        reader = csv.reader(file)
        lines = list(reader)  

        # Modify the last line
        if lines:  
            lines[-1].extend([elapsed_time2, memory_used2])

        # Move cursor to the start & clear file before writing
        file.seek(0)
        file.truncate()

        writer = csv.writer(file) 
        writer.writerows(lines)
    #process_prompt_file(cfg.prompts_dir.directory, cfg)

if __name__ == "__main__":
    main()