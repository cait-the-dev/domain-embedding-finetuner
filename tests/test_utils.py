from src.utils import chunk_text

SENTENCE = "alpha bravo charlie delta echo foxtrot golf hotel"


def test_chunk_text_overlap():
    text = (" " + SENTENCE) * 120
    chunks = chunk_text(text, max_tokens=100, overlap=0.25)

    assert all(80 <= len(c.split()) <= 120 for c in chunks)

    for a, b in zip(chunks, chunks[1:]):
        overlap = len(set(a.split()) & set(b.split()))
        assert overlap >= 5
