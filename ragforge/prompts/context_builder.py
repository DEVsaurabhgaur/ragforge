from typing import List
from ragforge.models.chunk import DocumentChunk

def build_prompt_context(chunks: List[DocumentChunk]) -> str:
    return '\n---\n'.join([f'[Source {i+1}]: {c.text}' for i, c in enumerate(chunks)])
