import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from src import evaluate, generate, ingest

TMP = Path(tempfile.mkdtemp())


def tiny_cfg():
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    cfg["paths"]["pdf_chunks_jsonl"] = str(TMP / "chunks.jsonl")
    cfg["paths"]["train_jsonl"] = str(TMP / "train.jsonl")
    cfg["paths"]["finetuned_gguf"] = "tests/fixtures/dummy.gguf"
    cfg["ingest"]["max_tokens"] = 50
    cfg["ingest"]["overlap"] = 0.1
    cfg["generate"]["batch_size"] = 1
    return cfg


def test_smoke_pipeline(monkeypatch):
    cfg = tiny_cfg()
    cfg_path = TMP / "tiny_cfg.yaml"
    yaml.safe_dump(cfg, cfg_path.open("w"))

    # 1. ingest (first 1 page only)
    monkeypatch.setattr(ingest, "NUM_PAGES_LIMIT", 1, raising=False)
    ingest.main(str(cfg_path))
    assert Path(cfg["paths"]["pdf_chunks_jsonl"]).exists()

    # 2. generate (mock GPT)
    def fake_chat(*_, **__):
        txt = "Q: test?\nA: stub."
        return MagicMock(choices=[MagicMock(message=MagicMock(content=txt))])

    with patch("openai.resources.chat.Completions.create", side_effect=fake_chat):
        generate.main(str(cfg_path))
    assert Path(cfg["paths"]["train_jsonl"]).exists()

    # 3. evaluate retrieval only (skip llama & judge)
    with patch.object(evaluate, "load_llama", lambda *a, **k: None):
        with patch.object(evaluate, "answer_with_llama", lambda *a, **k: "stub"):
            with patch.object(evaluate, "judge_answers_openai", lambda x: [1] * len(x)):
                evaluate.evaluate(str(cfg_path))
