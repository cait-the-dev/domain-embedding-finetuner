from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

try:
    import faiss

    HAVE_FAISS = True
except ModuleNotFoundError:
    from sklearn.neighbors import NearestNeighbors

    HAVE_FAISS = False

from llama_cpp import Llama


def load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def build_index(texts: List[str], model: SentenceTransformer):
    vecs = (
        model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        .astype("float32")
        .copy()
    )

    if HAVE_FAISS:
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        knn = lambda q_vec, k: index.search(np.array([q_vec], dtype="float32"), k)[1][0]
    else:
        nn = NearestNeighbors(metric="cosine").fit(vecs)
        knn = lambda q_vec, k: nn.kneighbors(
            [q_vec], n_neighbors=k, return_distance=False
        )[0]

    return knn


def build_prompt(context: List[str], question: str) -> str:
    return (
        "### Context\n"
        + "\n\n".join(context)
        + f"\n\n### Question\n{question}\n\n### Answer (concise):"
    )


def main(cfg_path="config.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]
    demo_cfg = cfg.get("demo", {})
    top_k = demo_cfg.get("top_k", 3)
    n_threads = demo_cfg.get("n_threads", 8)

    chunks = load_jsonl(paths["pdf_chunks_jsonl"])
    chunk_texts = [c["text"] for c in chunks]
    logging.info("Building MiniLM index on %d chunks …", len(chunk_texts))
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    knn = build_index(chunk_texts, st_model)

    logging.info("Loading llama.cpp model … (~4-5 s on CPU)")
    llm = Llama(
        model_path=paths["finetuned_gguf"],
        n_threads=n_threads,
        n_gpu_layers=0,
        logits_all=False,
        verbose=False,
    )

    print("\nRAG demo ready. Type a question (or just press <enter> to quit):\n")

    while True:
        question = input("> ").strip()
        if not question:
            break

        t0 = time.time()
        q_vec = st_model.encode(question, normalize_embeddings=True)
        idx = knn(q_vec, top_k)
        context = [chunk_texts[i] for i in idx]
        prompt = build_prompt(context, question)

        out = llm(
            prompt,
            max_tokens=128,
            temperature=0.2,
            repeat_penalty=1.1,
            stop=["###"],
        )
        answer = out["choices"][0]["text"].strip()
        dt = time.time() - t0

        print(f"\n{answer}\n--- ({dt*1000:.0f} ms)\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(parser.parse_args().config)
