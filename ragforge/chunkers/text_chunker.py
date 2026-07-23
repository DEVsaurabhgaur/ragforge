from typing import List
from ragforge.models.chunk import DocumentChunk
import uuid

def recursive_character_chunker(text: str, chunk_size: int = 512, overlap: int = 64) -> List[DocumentChunk]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        sub = text[start:end]
        chunks.append(DocumentChunk(chunk_id=str(uuid.uuid4()), text=sub))
        start += chunk_size - overlap
    return chunks
