from dataclasses import dataclass
from ragforge.models.chunk import DocumentChunk

@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
