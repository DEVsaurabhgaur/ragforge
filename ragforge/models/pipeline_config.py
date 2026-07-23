from dataclasses import dataclass

@dataclass
class RagConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    vector_store: str = 'faiss'
    embedding_model: str = 'text-embedding-3-small'
