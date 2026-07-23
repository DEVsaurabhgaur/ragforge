from dataclasses import dataclass
from typing import List

@dataclass
class PromptContext:
    formatted_chunks: str
    source_citations: List[str]
