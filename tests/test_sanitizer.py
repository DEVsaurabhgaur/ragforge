from ragforge.utils.sanitizer import clean_markdown_text
def test_clean():
    assert clean_markdown_text('# Header') == 'Header'
