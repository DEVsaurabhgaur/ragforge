from ragforge.embedders.mock_embedder import get_mock_embedding
def test_embed():
    vec = get_mock_embedding(128)
    assert len(vec) == 128
