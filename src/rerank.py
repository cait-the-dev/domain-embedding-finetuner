from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_cross_encoder(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    *,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Loading cross-encoder %s on %s …", model_name, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device).eval()

    return tokenizer, model, device


@torch.inference_mode()
def rerank(
    query: str,
    passages: List[str],
    *,
    tokenizer,
    model,
    device: str,
    top_n: int = 3,
) -> List[int]:
    if not passages:
        return []

    toks = tokenizer(
        [[query, p] for p in passages],
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    logits = model(**toks).logits.squeeze(-1)
    scores = logits.cpu().numpy()

    top_n = min(top_n, len(passages))
    ranked = np.argsort(-scores)[:top_n]

    logging.debug("Re-ranker scores: %s", scores.tolist())
    return ranked.tolist()
