from __future__ import annotations

import sys
import types
from typing import Dict, List

import numpy as np
import pytest
import torch

import src.retrieval as retrieval
from src.evaluate import _hits as hits
from src.evaluate import _llama_embed as llama_embed
from src.evaluate import _load_llama as load_llama
from src.evaluate import _uplift as uplift
from src.rerank import rerank
from src.retrieval import Retriever
from src.utils import dynamic_gpu_layers


class _TinyST:
    def __init__(self, *_, **__): ...
    def encode(self, xs, *, normalize_embeddings=False, **__):
        if isinstance(xs, str):
            xs = [xs]
        v = np.asarray([[len(t), 1.0] for t in xs], dtype="float32")
        if normalize_embeddings:
            v = v / np.linalg.norm(v, axis=1, keepdims=True)
        return v if len(xs) > 1 else v[0]


class _TinyCE(_TinyST):
    def predict(self, pairs, **__):
        return np.asarray(
            [float(len(set(q.split()) & set(d.split()))) for q, d in pairs],
            dtype="float32",
        )


@pytest.fixture(scope="module")
def passages() -> List[Dict[str, str]]:
    return [
        {"text": "Paris is the capital of France."},
        {"text": "Berlin is the capital of Germany."},
        {"text": "Tokyo is in Japan."},
    ]


@pytest.fixture(scope="module")
def qa_set():
    return [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Which city is Germany's capital?", "answer": "Berlin"},
    ]


def test_retrieve_hits(monkeypatch, passages, qa_set):
    monkeypatch.setattr(retrieval, "SentenceTransformer", _TinyST, raising=True)
    monkeypatch.setattr(retrieval, "CrossEncoder", _TinyCE, raising=True)

    ret = Retriever(device="cuda").fit(passages)

    rate = hits(
        qa_set,
        [p["text"] for p in passages],
        knn=ret._knn,
        q_embed=lambda q: _TinyST().encode(q, normalize_embeddings=True),
        k=1,
    )
    assert rate == 1.0


def test_uplift_math():
    d_pp, d_pct = uplift(0.40, 0.70)
    assert abs(d_pp - 0.30) < 1e-6
    assert abs(d_pct - 75.0) < 1e-6


def test_rerank_simple():
    query = "capital france"
    docs = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "Tokyo is in Japan.",
    ]

    class _Batch(dict):
        def to(self, *_):
            return self

    def _tok(pairs, **__):
        return _Batch(
            input_ids=np.zeros((len(pairs), 5), dtype="int64"),
            attention_mask=np.ones((len(pairs), 5), dtype="int64"),
        )

    class _DummyModel:
        def eval(self): ...
        def to(self, *_):
            return self

        def __call__(self, **__):
            scores = torch.arange(len(docs) - 1, -1, -1, dtype=torch.float32)[
                :, None
            ]
            return types.SimpleNamespace(logits=scores)

    best = rerank(
        query,
        docs,
        tokenizer=_tok,
        model=_DummyModel(),
        device="cuda",
        top_n=2,
    )
    top_idx = best[0]
    assert "Paris" in docs[top_idx]


class _FakeLlama:
    def embed(self, txt: str):
        return np.asarray([len(txt), ord(txt[0]), ord(txt[-1])], dtype="float32")

    def __call__(self, prompt: str, **__):
        return {"choices": [{"text": f"echo:{prompt.splitlines()[-1]}"}]}


def test_load_llama_and_helpers(monkeypatch):
    stub = types.ModuleType("llama_cpp")
    stub.Llama = lambda *_, **__: _FakeLlama()
    monkeypatch.setitem(sys.modules, "llama_cpp", stub)

    llm = load_llama("dummy.gguf", threads=2, embed=True, ngl=16)

    vec = llama_embed(llm, "abc")
    assert vec.tolist() == [3.0, float(ord("a")), float(ord("c"))]

    assert llm("Q?\nA:")["choices"][0]["text"].startswith("echo:")
    assert 0 <= dynamic_gpu_layers(16) <= 16
