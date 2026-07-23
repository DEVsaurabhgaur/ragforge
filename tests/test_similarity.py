from ragforge.retrievers.similarity import cosine_similarity
def test_sim():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0
