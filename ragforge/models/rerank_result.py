from dataclasses import dataclass
from ragforge.models.search_result import SearchResult

@dataclass
class RerankedResult:
    result: SearchResult
    rerank_score: float
