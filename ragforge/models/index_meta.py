from dataclasses import dataclass
import time

@dataclass
class IndexMetadata:
    index_name: str
    total_vectors: int
    created_at: float = time.time()
