from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding_vector: list = field(default_factory=list)
