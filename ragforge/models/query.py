from dataclasses import dataclass
from typing import Optional, List

@dataclass
class RetrievalQuery:
    query_text: str
    top_k: int = 5
    filters: Optional[dict] = None
