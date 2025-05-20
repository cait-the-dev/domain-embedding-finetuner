from unittest.mock import MagicMock, patch

from src.generate import _create_batches, _qa_from_response, generate_qa_pairs

DUMMY_CHUNKS = [{"chunk_id": "1", "text": "The sky is blue."}]


def fake_chat_completion(*_, **__):
    return MagicMock(
        choices=[
            MagicMock(message=MagicMock(content="Q: What color is the sky?\nA: blue."))
        ]
    )


@patch("openai.resources.chat.Completions.create", side_effect=fake_chat_completion)
def test_generate_filters(mock_openai):
    qa_pairs = generate_qa_pairs(DUMMY_CHUNKS, batch_size=1)

    assert len(qa_pairs) == 1
    q, a = qa_pairs[0]["question"], qa_pairs[0]["answer"]
    assert "color" in q.lower() and "blue" in a.lower()
