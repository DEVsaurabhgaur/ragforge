from ragforge.chunkers.text_chunker import recursive_character_chunker
def test_chunker():
    text = 'A' * 1000
    chunks = recursive_character_chunker(text, 500, 50)
    assert len(chunks) >= 2
