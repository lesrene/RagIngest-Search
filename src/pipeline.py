import hydra
import subprocess
from omegaconf import DictConfig

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n🚀 Running ingestion process with the following settings:")
    print(f"Chunk Size: {cfg.chunk_size}")
    print(f"Chunk Overlap: {cfg.chunk_overlap}")
    print(f"Vector Database: {cfg.vector_db}")
    print(f"Embedding Model: {cfg.embedding_model}")
    print(f"LLM: {cfg.llm}\n")

    # Run ingestion script
    subprocess.run(["python", cfg.scripts.injest, 
                    f"--chunk_size={cfg.chunk_size}", 
                    f"--chunk_overlap={cfg.chunk_overlap}", 
                    f"--vector_db={cfg.vector_db}", 
                    f"--embedding_model={cfg.embedding_model}", 
                    f"--llm={cfg.llm}"], check=True)

    print("\n🔍 Running search process...")
    
    # Run search script
    subprocess.run(["python", cfg.scripts.search,
                    f"--vector_db={cfg.vector_db}",
                    f"--embedding_model={cfg.embedding_model}",
                    f"--llm={cfg.llm}"], check=True)

    print("\n✅ Pipeline execution completed.")

if __name__ == "__main__":
    main()
