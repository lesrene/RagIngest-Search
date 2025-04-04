import hydra
from omegaconf import DictConfig
import ollama
import redis
from redis.commands.search.query import Query
import chromadb
import faiss
import numpy as np
import os
import fitz
import re
import pandas as pd
import pdfplumber
from sentence_transformers import SentenceTransformer
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
    
class FAISSManager:
    def __init__(self, dim, index_path="/Users/lesrene/Desktop/faiss_index.bin"):
        self.index_path = index_path
        self.keys = []
        
        if os.path.exists(self.index_path):
            print(f"Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(dim)
            print(f"Creating new FAISS index at {self.index_path}")

    def add_embedding(self, embedding, key):
        self.index.add(np.array([embedding], dtype=np.float32))
        self.keys.append(key)
        self.save_index()  # Save after adding

    def save_index(self):
        faiss.write_index(self.index, self.index_path)
        print(f"FAISS index saved to {self.index_path}")

    def search(self, embedding, k=5):
        D, I = self.index.search(np.array([embedding], dtype=np.float32), k)
        return [(self.keys[i], D[0][idx]) for idx, i in enumerate(I[0]) if i != -1]

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page

def preprocess_text(text: str) -> str:
    """Preprocess text by removing extra whitespace, punctuation, and other noise."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\*\-\•]', '', text)
    return text.strip()

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF file and return as a list of DataFrames."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                df = pd.DataFrame(table)
                tables.append(df)
    return tables

def process_pdfs(cfg, split_text_into_chunks, get_embedding, store_embedding):
    #print(cfg)
    for file_name in os.listdir(cfg.data_dir.directory):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(cfg.data_dir.directory, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            tables = extract_tables_from_pdf(pdf_path)
            
            for page_num, text in text_by_page:
                cleaned_text = preprocess_text(text)
                chunks = split_text_into_chunks(cleaned_text, cfg.chunk_size, cfg.chunk_overlap)

                for chunk in chunks:
                    embedding = get_embedding(chunk, cfg)
                    store_embedding(file_name, str(page_num), chunk, embedding)
                
            print(f"Processed {file_name} (Tables: {len(tables)})")

@hydra.main(config_path="/Users/lesrene/Desktop/DS4300/RagIngestAndSearch/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Chunk Size: {cfg.chunk_size}")
    print(f"Chunk Overlap: {cfg.chunk_overlap}")
    print(f"Vector Database: {cfg.vector_db.name}")  
    print(f"Embedding Model: {cfg.embedding_model.name}")
    print(f"LLM: {cfg.llm.name}\n")
    # Initialize vector DB
    if cfg.vector_db.name == "redis":
        redis_client = redis.Redis(host=cfg.vector_db.host, port=cfg.vector_db.port, db=cfg.vector_db.db)

    elif cfg.vector_db.name == "chroma":
        chroma_client = chromadb.HttpClient(host=cfg.vector_db.host, port=cfg.vector_db.port)
        chroma_collection = chroma_client.get_or_create_collection(name=cfg.vector_db.collection_name)
        print(cfg.vector_db.host, cfg.vector_db.port)
        #chroma_client = chromadb.HttpClient(host="localhost", port=8000)
        #chroma_collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

    elif cfg.vector_db.name == "faiss":
        faiss_index = FAISSManager(cfg.embedding_model.vector_dim)

    def get_embedding(text: str, cfg) -> list:
        if cfg.embedding_model.name == "nomic-embed-text":
            response = ollama.embeddings(model=cfg.embedding_model.name, prompt=text)
            return response["embedding"]
        elif cfg.embedding_model.name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
            model = SentenceTransformer(cfg.embedding_model.name)
            return model.encode(text).tolist()
        else:
            raise ValueError(f"Unsupported embedding model: {cfg.embedding_model.name}")

    def store_embedding(file: str, page: str, chunk: str, embedding: list):
        key = f"{cfg.vector_db.doc_prefix}:{file}_page_{page}"

        if cfg.vector_db.name == "redis":
            redis_client.hset(
                key,
                mapping={"file": file, "page": page, "chunk": chunk, "embedding": np.array(embedding, dtype=np.float32).tobytes()}
            )

        elif cfg.vector_db.name == "faiss":
            faiss_index.add_embedding(embedding, key)

        elif cfg.vector_db.name == "chroma":
            chroma_collection.add(
                ids=[key],
                embeddings=[embedding],
                metadatas={"file": file, "page": page, "chunk": chunk}
            )

        print(f"Stored embedding for: {chunk}")

    def split_text_into_chunks(text, chunk_size, overlap):
        words = text.split()
        chunks = [" ".join(words[i: i + chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
        return chunks

    result1, elapsed_time1, memory_used1 = track_performance(process_pdfs, cfg, split_text_into_chunks, get_embedding, store_embedding)
    print("\n---Done processing PDFs---\n")

    with open('performance_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([cfg.vector_db.name, cfg.llm.name, cfg.embedding_model.name, cfg.chunk_size, cfg.chunk_overlap, elapsed_time1, memory_used1])

    def query_vdb(query_text: str):
        if cfg.vector_db.name == "redis":
            print("\n🔎 Querying Redis...")
            q = (
                Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
                .sort_by("vector_distance")
                .return_fields("id", "vector_distance")
                .dialect(2)
            )
            query_text = "Efficient search in vector databases"
            embedding = get_embedding(query_text, cfg)
            res = redis_client.ft(cfg.vector_db.index_name).search(
                q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
            )
            # print(res.docs)

            for doc in res.docs:
                print(f"{doc.id} \n ----> {doc.vector_distance}\n")

        elif cfg.vector_db.name == "faiss":
            print("\n🔎 Querying FAISS...")
            embedding = get_embedding(query_text, cfg)
            faiss_results = faiss_index.search(embedding, k=5)
            for key, distance in faiss_results:
                print(f"{key} \n ----> Distance: {distance}\n")

        elif cfg.vector_db.name == "chroma":
            print("\n🔎 Querying ChromaDB...")
            embedding = get_embedding(query_text, cfg)
            chroma_results = chroma_collection.query(query_embeddings=[embedding], n_results=5)
            for doc_id, score in zip(chroma_results["ids"][0], chroma_results["distances"][0]):
                print(f"{doc_id} \n ----> Distance: {score}\n")

    query_vdb("What is the capital of France?")

if __name__ == "__main__":
    main()