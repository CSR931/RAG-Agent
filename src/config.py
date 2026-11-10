import os
from dotenv import load_dotenv
load_dotenv()

def load_config():
    return {
        "chunk_size": int(os.getenv("CHUNK_SIZE", 400)),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", 80)),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "top_k": int(os.getenv("TOP_K", 5)),
    }
