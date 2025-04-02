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

# Function to generate response
def generate_rag_response(query, context_results, cfg):
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    response = ollama.chat(
        model=cfg.llm.model, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


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

        print(responses)

    except Exception as e:
        print(f" Error processing file: {e}")

@hydra.main(config_path="/Users/lesrene/Desktop/DS4300/RagIngestAndSearch/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    #file_path = "prompts.txt"  # Change this to your actual file path
    file_path = cfg.prompts_dir.directory
    process_prompt_file(file_path, cfg)
    #process_prompt_file(cfg.prompts_dir.directory, cfg)

if __name__ == "__main__":
    main()