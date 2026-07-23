from ragforge.pipeline.rag_engine import RagEngine
from ragforge.models.pipeline_config import RagConfig
def test_engine():
    e = RagEngine(RagConfig())
    assert e.config.chunk_size == 512
