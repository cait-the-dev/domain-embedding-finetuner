from __future__ import annotations

import logging
import os
from typing import Callable, Dict, List

import numpy as np

try:
    import faiss

    _HAVE_FAISS = True
except ModuleNotFoundError:
    from sklearn.neighbors import NearestNeighbors

    _HAVE_FAISS = False

from sentence_transformers import CrossEncoder, SentenceTransformer

from src.utils import num_tokens

DEFAULT_EMBED = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANK = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _fit_knn(vecs: np.ndarray) -> Callable[[np.ndarray, int], List[int]]:
    if _HAVE_FAISS:
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        return lambda q, k: idx.search(np.array([q], dtype="float32"), k)[1][
            0
        ].tolist()

    nn = NearestNeighbors(metric="cosine").fit(vecs)
    return lambda q, k: nn.kneighbors([q], n_neighbors=k, return_distance=False)[
        0
    ].tolist()


class Retriever:
    def __init__(
        self,
        bi_encoder: str | SentenceTransformer = DEFAULT_EMBED,
        cross_encoder: str | CrossEncoder = DEFAULT_RERANK,
        device: str | None = None,
    ):
        dev = device or ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")

        self.bi: SentenceTransformer = (
            bi_encoder
            if isinstance(bi_encoder, SentenceTransformer)
            else SentenceTransformer(bi_encoder, device=dev)
        )
        self.ce: CrossEncoder = (
            cross_encoder
            if isinstance(cross_encoder, CrossEncoder)
            else CrossEncoder(cross_encoder, device=dev)
        )

        self._texts: List[str] = []
        self._embs: np.ndarray | None = None
        self._knn: Callable[[np.ndarray, int], List[int]] | None = None

    def fit(self, chunks: List[Dict[str, str]]) -> "Retriever":
        self._texts = [c["text"] if isinstance(c, dict) else c for c in chunks]
        self._embs = (
            self.bi.encode(
                self._texts, show_progress_bar=True, normalize_embeddings=True
            )
            .astype("float32")
            .copy()
        )
        self._knn = _fit_knn(self._embs)
        logging.info("Retriever fitted on %d passages", len(self._texts))
        return self

    def __call__(
        self,
        question: str,
        *,
        k: int = 48,
        max_ctx_tokens: int = 2048,
        merge_adjacent: bool = True,
    ) -> List[str]:
        if self._knn is None:
            raise RuntimeError("Retriever.fit() must be called first")

        q_vec = self.bi.encode(question, normalize_embeddings=True)
        cand_idx = self._knn(q_vec, k=k)

        scores = self.ce.predict(
            [[question, self._texts[i]] for i in cand_idx], convert_to_numpy=True
        )
        order = np.argsort(-scores)

        packed: List[str] = []
        used = 0
        for j in order:
            txt = self._texts[cand_idx[j]]

            if merge_adjacent and packed and "\f" not in txt:
                prev = packed[-1]
                if len(prev) < 60 and len(txt) < 200:
                    txt = prev + " " + txt
                    used -= num_tokens(prev)
                    packed.pop()

            tks = num_tokens(txt)
            if used + tks > max_ctx_tokens:
                continue
            packed.append(txt)
            used += tks

        return packed
