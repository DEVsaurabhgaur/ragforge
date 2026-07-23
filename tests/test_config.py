from ragforge.models.pipeline_config import RagConfig
def test_config():
    c = RagConfig()
    assert c.chunk_size == 512
