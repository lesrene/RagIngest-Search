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

# Function to get embedding
def get_embedding(text: str, model: str) -> list:
    """Generate embeddings using the specified model."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# Function to search in Redis
def search_redis(query, cfg, top_k=3):
    redis_client = redis.StrictRedis(host=cfg.redis.host, port=cfg.redis.port, decode_responses=True)

    query_embedding = get_embedding(query, model=cfg.embedding_model.name)
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

        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        return top_results

    except Exception as e:
        print(f"Redis search error: {e}")
        return []

# Function to search in FAISS
def search_faiss(query, cfg, top_k=3):
    index = faiss.read_index(cfg.faiss.index_path)
    
    query_embedding = np.array(get_embedding(query, model=cfg.embedding_model.name), dtype=np.float32)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Reshape for FAISS

    distances, indices = index.search(query_embedding, top_k)

    return [{"index": int(idx), "similarity": float(dist)} for idx, dist in zip(indices[0], distances[0]) if idx != -1]

# Function to search in ChromaDB
def search_chroma(query, cfg, top_k=3):
    client = chromadb.PersistentClient(path=cfg.chroma.db_path)
    collection = client.get_or_create_collection(cfg.chroma.collection_name)

    query_embedding = get_embedding(query, model=cfg.embedding_model.name)
    
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
        model=cfg.llm.name, messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# Interactive search
def interactive_search(cfg):
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        context_results = search_embeddings(query, cfg)
        response = generate_rag_response(query, context_results, cfg)

        print("\n--- Response ---")
        print(response)

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    interactive_search(cfg)

if __name__ == "__main__":
    main()