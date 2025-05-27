from __future__ import annotations

import json
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from src import evaluate, generate

TMP = Path(tempfile.mkdtemp())


def tiny_cfg() -> dict:
    cfg = yaml.safe_load(Path("config.yaml").read_text())

    cfg["paths"]["pdf_chunks_jsonl"] = str(TMP / "chunks.jsonl")
    cfg["paths"]["train_jsonl"] = str(TMP / "train.jsonl")
    cfg["paths"]["finetuned_gguf"] = "tests/fixtures/dummy.gguf"
    cfg["paths"]["eval_manual_jsonl"] = str(TMP / "eval_manual.jsonl")
    cfg["paths"]["eval_holdout_jsonl"] = str(TMP / "eval_holdout.jsonl")

    cfg.setdefault("generate", {})["batch_size"] = 1
    cfg["generate"]["max_rows"] = 3
    return cfg


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_smoke_pipeline():
    cfg = tiny_cfg()
    cfg_path = TMP / "tiny_cfg.yaml"
    yaml.safe_dump(cfg, cfg_path.open("w"))

    _write_jsonl(
        Path(cfg["paths"]["pdf_chunks_jsonl"]),
        [{"chunk_id": 0, "text": "Dummy passage."}],
    )
    _write_jsonl(
        Path(cfg["paths"]["eval_manual_jsonl"]), [{"question": "Q?", "answer": "A"}]
    )
    _write_jsonl(
        Path(cfg["paths"]["eval_holdout_jsonl"]),
        [{"question": "Q2?", "answer": "A2"}],
    )

    def _fake_chat(*_, **__):
        txt = json.dumps({"question": "stub-q?", "answer": "stub-a"})
        return MagicMock(choices=[MagicMock(message=MagicMock(content=txt))])

    with patch("src.generate._openai_chat", side_effect=_fake_chat):
        generate.main(str(cfg_path))

    assert Path(cfg["paths"]["train_jsonl"]).exists()

    class _DummyRet(types.SimpleNamespace):
        def __init__(self):
            super().__init__()
            self._knn = lambda _v, k=3: [0] * k

        def fit(self, *_a, **_k):
            return self

        def __call__(self, *_a, **_k):
            return ["ctx"]

    class _DummyLLM:
        D = 384

        def __call__(self, *_, **__):
            return {"choices": [{"text": "stub"}]}

        def embed(self, _txt):
            return np.zeros(self.D, dtype="float32")

    DummyRetrieverFactory = lambda *a, **k: _DummyRet()
    D = _DummyLLM.D

    with patch.object(
        evaluate, "Retriever", DummyRetrieverFactory, create=True
    ), patch.object(
        evaluate, "_load_llama", lambda *a, **k: _DummyLLM(), create=True
    ), patch.object(
        evaluate,
        "_llama_embed",
        lambda *a, **k: np.zeros(D, dtype="float32"),
        create=True,
    ), patch.object(
        evaluate,
        "judge_answers_openai",
        lambda items, **_: [1] * len(items),
        create=True,
    ):
        evaluate.evaluate(str(cfg_path))
