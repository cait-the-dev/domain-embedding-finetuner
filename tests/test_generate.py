import json
from unittest.mock import MagicMock, patch

from src.generate import generate_qa_pairs

DUMMY_CHUNKS = [{"chunk_id": "1", "text": "The sky is blue."}]


def fake_chat_completion(*_, **__):
    obj = {"question": "What color is the sky?", "answer": "It is usually blue."}
    return MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(obj)))])


@patch("src.generate._openai_chat", side_effect=fake_chat_completion)
def test_generate_filters(mock_openai):
    qa_pairs = generate_qa_pairs(DUMMY_CHUNKS, batch_size=1, max_rows=1)
    assert len(qa_pairs) == 1
    q, a = qa_pairs[0]["question"], qa_pairs[0]["answer"]
    assert "color" in q.lower() and "blue" in a.lower()
