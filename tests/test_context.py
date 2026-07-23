from ragforge.prompts.context_builder import build_prompt_context
from ragforge.models.chunk import DocumentChunk
def test_context():
    ctx = build_prompt_context([DocumentChunk('1', 'sample text')])
    assert 'sample text' in ctx
