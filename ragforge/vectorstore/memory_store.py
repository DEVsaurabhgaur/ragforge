from typing import List
from ragforge.models.search_result import SearchResult
from ragforge.models.chunk import DocumentChunk
from ragforge.retrievers.similarity import cosine_similarity

class MemoryVectorStore:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
    def add_chunk(self, chunk: DocumentChunk):
        self.chunks.append(chunk)
    def search(self, query_vec: list, top_k: int = 5) -> List[SearchResult]:
        results = []
        for c in self.chunks:
            score = cosine_similarity(query_vec, c.embedding_vector)
            results.append(SearchResult(chunk=c, score=score))
        return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
