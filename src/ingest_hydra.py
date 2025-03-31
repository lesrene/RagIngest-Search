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

class FAISSManager:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.keys = []

    def add_embedding(self, embedding, key):
        self.index.add(np.array([embedding], dtype=np.float32))
        self.keys.append(key)

    # search FAISS index for the nearest neighbors of the given embedding
    def search(self, embedding, k=5):
        D, I = self.index.search(np.array([embedding], dtype=np.float32), k)
        return [(self.keys[i], D[0][idx]) for idx, i in enumerate(I[0]) if i != -1]
    

# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


def preprocess_text(text: str) -> str:
    """Preprocess text by removing extra whitespace, punctuation, and other noise."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\*\-\•]', '', text)  # Keep bullet characters
    return text.strip()

def extract_tables_from_pdf(pdf_path):
    """Extract tables from a PDF file and return as a list of DataFrames."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                # Convert to DataFrame for structured storage
                df = pd.DataFrame(table)
                tables.append(df)
    return tables

def detect_bullet_points(text):
    """Identify and format bullet points and numbered lists in text."""
    bullet_point_patterns = [
        r'^\s*[\*\-•]\s+',   # Match *, -, • followed by space
        r'^\s*\d+\.\s+',     # Match numbered lists like 1., 2., 3.
        r'^\s*[a-zA-Z]\)\s+' # Match lettered lists like a), b), c)
    ]
    
    formatted_lines = []
    for line in text.split("\n"):
        if any(re.match(pattern, line) for pattern in bullet_point_patterns):
            formatted_lines.append(f"- {line.strip()}")  # Convert to a standardized bullet format
        else:
            formatted_lines.append(line.strip())
    
    return "\n".join(formatted_lines)

def extract_captions_from_text(text):
    """Identify possible captions in extracted text."""
    caption_patterns = [
        r'Figure\s\d+[:\.\-]',  # Matches "Figure 1:", "Figure 2.", "Figure 3 -"
        r'Table\s\d+[:\.\-]',   # Matches "Table 1:", "Table 2."
    ]
    
    captions = []
    for line in text.split("\n"):
        if any(re.search(pattern, line, re.IGNORECASE) for pattern in caption_patterns):
            captions.append(line.strip())
    
    return captions


def process_pdfs(data_dir):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)  # Extract text from PDF
            tables = extract_tables_from_pdf(pdf_path)  # Extract tables separately
            
            for page_num, text in text_by_page:
                cleaned_text = preprocess_text(text)  # Clean text first
                formatted_text = detect_bullet_points(cleaned_text)  # Then format bullet points
                captions = extract_captions_from_text(formatted_text)  # Extract captions
                chunks = split_text_into_chunks(formatted_text)  # Chunk the cleaned, formatted text
                # print(f"  Chunks: {chunks}")
                
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)  
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
                
            print(f"-----> Processed {file_name} (Tables: {len(tables)})")

    
@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # Initialize vector DB
    if cfg.vector_db.name == "redis":
        redis_client = redis.Redis(host=cfg.vector_db.host, port=cfg.vector_db.port, db=cfg.vector_db.db)

        def create_hnsw_index():
            try:
                redis_client.execute_command(f"FT.DROPINDEX {cfg.vector_db.index_name} DD")
            except redis.exceptions.ResponseError:
                pass

            redis_client.execute_command(
                f"""
                FT.CREATE {cfg.vector_db.index_name} ON HASH PREFIX 1 {cfg.vector_db.doc_prefix}
                SCHEMA text TEXT
                embedding VECTOR HNSW 6 DIM {cfg.vector_db.vector_dim} TYPE FLOAT32 DISTANCE_METRIC {cfg.vector_db.distance_metric}
                """
            )
            print("Index created successfully.")

        create_hnsw_index()

    elif cfg.vector_db.name == "chroma":
        chroma_client = chromadb.HttpClient(host=cfg.vector_db.host, port=cfg.vector_db.port)
        chroma_collection = chroma_client.get_or_create_collection(name="pdf_embeddings")

    elif cfg.vector_db.name == "faiss":
        #faiss_index = FAISSManager(VECTOR_DIM)
        faiss_index = FAISSManager(cfg.vector_db.vector_dim)

    def clear_store():
        print("Clearing existing vector stores...")
        if cfg.vector_db.name == "redis":
            redis_client.flushdb()
        elif cfg.vector_db.name == "faiss":
            faiss_index.index.reset() 
        elif cfg.vector_db.name == "chroma":
            chroma_collection.delete(where={"$exists": True})
        print("All stores cleared.")

    clear_store()

    def get_embedding(text: str) -> list:
        if cfg.embedding_model.name == "nomic":
            response = ollama.embeddings(model=cfg.embedding_model.name, prompt=text)
        return response["embedding"]

    def store_embedding(file: str, page: str, chunk: str, embedding: list):
        key = f"{cfg.vector_db.doc_prefix}:{file}_page_{page}_chunk_{chunk}"

        if cfg.vector_db.name == "redis":
            redis_client.hset(
                key,
                mapping={
                    "file": file,
                    "page": page,
                    "chunk": chunk,
                    "embedding": np.array(embedding, dtype=np.float32).tobytes(),
                },
            )

        elif cfg.vector_db.name == "faiss":
            faiss_index.add_embedding(embedding, key)

        elif cfg.vector_db.name == "chroma":
            chroma_collection.add(
                ids=[key],
                embeddings=[embedding],
                metadatas={"file": file, "page": page, "chunk": chunk}
            )

        print(f" Stored embedding for: {chunk}")

    def split_text_into_chunks(text, chunk_size=cfg.chunking.chunk_size, overlap=cfg.chunking.chunk_overlap):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i: i + chunk_size])
            chunks.append(chunk)
        return chunks

    process_pdfs(cfg.data.directory)
    print("\n---Done processing PDFs---\n")

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
            embedding = get_embedding(query_text)
            res = redis_client.ft(cfg.vector_db.index_name).search(
                q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
            )
            # print(res.docs)

            for doc in res.docs:
                print(f"{doc.id} \n ----> {doc.vector_distance}\n")

        elif cfg.vector_db.name == "faiss":
            print("\n🔎 Querying FAISS...")
            faiss_results = faiss_index.search(embedding, k=5)
            for key, distance in faiss_results:
                print(f"{key} \n ----> Distance: {distance}\n")

        elif cfg.vector_db.name == "chroma":
            print("\n🔎 Querying ChromaDB...")
            chroma_results = chroma_collection.query(query_embeddings=[embedding], n_results=5)
            for doc_id, score in zip(chroma_results["ids"][0], chroma_results["distances"][0]):
                print(f"{doc_id} \n ----> Distance: {score}\n")

    query_vdb("What is the capital of France?")
    


if __name__ == "__main__":
    main()