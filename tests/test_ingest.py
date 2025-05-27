from src.utils import chunk_text


def test_chunk_window_and_min_len():
    txt = "word " * 1000
    chunks = chunk_text(txt, max_tokens=100, overlap=0.2)
    assert all(80 <= len(c.split()) <= 120 for c in chunks)
    assert min(len(c.split()) for c in chunks) >= 40
