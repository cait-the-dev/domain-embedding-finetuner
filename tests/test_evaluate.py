from sentence_transformers import SentenceTransformer

from src.evaluate import retrieve_hits, uplift

st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

CHUNKS = [
    {"text": "Paris is the capital of France."},
    {"text": "Berlin is the capital of Germany."},
    {"text": "Tokyo is the capital of Japan."},
]

QA_SET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "Which city is Germany's capital?", "answer": "Berlin"},
]


def test_retrieve_hits():
    from src.evaluate import build_index

    index, _ = build_index(CHUNKS, st)
    hits = retrieve_hits(QA_SET, CHUNKS, st, index, k=1)
    assert hits == 1.0


def test_uplift_math():
    delta_pp, delta_pct = uplift(0.50, 0.80)
    assert abs(delta_pp - 0.30) < 1e-6
    assert abs(delta_pct - 60.0) < 1e-6
