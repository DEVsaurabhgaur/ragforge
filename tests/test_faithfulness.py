from ragforge.eval.faithfulness import check_faithfulness
def test_faith():
    assert check_faithfulness('hello', 'hello world') == 1.0
