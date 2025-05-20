from src.utils import chunk_text

SENTENCE = "alpha bravo charlie delta echo foxtrot golf hotel"


def test_chunk_text_overlap():
    text = (" " + SENTENCE) * 120  # ~960 tokens if we split on spaces
    chunks = chunk_text(text, max_tokens=100, overlap=0.25)

    # size window ≈ 100 tokens ± 20 %
    for c in chunks:
        n = len(c.split())
        assert 80 <= n <= 120, f"chunk length {n}"

    # successive windows should overlap by ≈25 tokens
    for a, b in zip(chunks, chunks[1:]):
        overlap = len(set(a.split()) & set(b.split()))
        assert 20 <= overlap <= 30
