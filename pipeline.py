# import hydra
# import subprocess
# from omegaconf import DictConfig

# @hydra.main(config_path="/Users/lesrene/Desktop/DS4300/RagIngestAndSearch/configs", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     print("\n🚀 Running ingestion process with the following settings:")
#     print(f"Chunk Size: {cfg.chunk_size}")
#     print(f"Chunk Overlap: {cfg.chunk_overlap}")
#     print(f"Vector Database: {cfg.vector_db.name}")  
#     print(f"Embedding Model: {cfg.embedding_model.name}")
#     print(f"LLM: {cfg.llm.name}\n")

#     # Run ingestion script (no CLI args needed)
#     subprocess.run(["python", cfg.scripts.injest], check=True)

#     print("\n🔍 Running search process...")

#     # Run search script (pass only necessary argument for prompts)
#     subprocess.run(["python", cfg.scripts.search], check=True)

#     print("\n✅ Pipeline execution completed.")

# if __name__ == "__main__":
#     main()

import hydra
import subprocess
import sys  # Needed to capture CLI args
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="/Users/lesrene/Desktop/DS4300/RagIngestAndSearch/configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n🚀 Running ingestion process with the following settings:")
    print(f"Chunk Size: {cfg.chunk_size}")
    print(f"Chunk Overlap: {cfg.chunk_overlap}")
    print(f"Vector Database: {cfg.vector_db.name}")  
    print(f"Embedding Model: {cfg.embedding_model.name}")
    print(f"LLM: {cfg.llm.name}\n")

    # Capture Hydra overrides passed in CLI
    hydra_overrides = sys.argv[1:]  # Extract everything after 'python pipeline.py'

    # Run ingestion script and forward Hydra overrides
    subprocess.run(["python", cfg.scripts.injest] + hydra_overrides, check=True)

    print("\n🔍 Running search process...")

    # Run search script and forward Hydra overrides
    subprocess.run(["python", cfg.scripts.search] + hydra_overrides, check=True)

    print("\n✅ Pipeline execution completed.")

if __name__ == "__main__":
    main()
