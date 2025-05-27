import json
import tempfile
from pathlib import Path

from src.finetune import make_dataset


class DummyTokenizer:
    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))[: kwargs.get("max_length", 128)]
        return {"input_ids": ids}


def test_make_dataset_tokenization():
    rows = [
        {"question": "What color is the sky?", "answer": "Blue."},
        {"question": "How many days in a week?", "answer": "Seven."},
    ]
    tmp = Path(tempfile.mkdtemp()) / "train.jsonl"
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    ds = make_dataset(str(tmp), DummyTokenizer(), max_len=128)

    assert len(ds) == 2
    for ex in ds:
        assert "input_ids" in ex and "labels" in ex
        assert ex["input_ids"] == ex["labels"]
        assert len(ex["input_ids"]) > 0 and len(ex["input_ids"]) <= 128
