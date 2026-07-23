from ragforge.vectorstore.memory_store import MemoryVectorStore
from ragforge.models.chunk import DocumentChunk
def test_store():
    store = MemoryVectorStore()
    store.add_chunk(DocumentChunk('1', 'text', embedding_vector=[1.0, 0.0]))
    res = store.search([1.0, 0.0], 1)
    assert len(res) == 1
