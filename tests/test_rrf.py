from ragforge.rerankers.rrf import reciprocal_rank_fusion
def test_rrf():
    res = reciprocal_rank_fusion(['d1', 'd2'], ['d2', 'd1'])
    assert 'd1' in res
