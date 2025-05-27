from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import yaml
from huggingface_hub import hf_hub_download
from torch import cuda
from tqdm import tqdm

from src.prompts import SYSTEM_JUDGE, build_judge_user, build_rag_prompt
from src.retrieval import Retriever
from src.utils import default_ctx, dynamic_gpu_layers


def _load_jsonl(p: str | Path) -> List[Dict]:
    with open(p, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def _ensure_base_gguf(
    path: str,
    repo_id: str = "QuantFactory/Llama-3.2-1B-GGUF",
    fn: str = "Llama-3.2-1B.Q8_0.gguf",
) -> str:
    tgt = Path(path)
    if tgt.exists():
        logging.info("Using cached baseline GGUF %s", tgt)
        return str(tgt)
    logging.info("Baseline GGUF missing – downloading …")
    hf_hub_download(
        repo_id=repo_id,
        filename=fn,
        local_dir=str(tgt.parent),
        local_dir_use_symlinks=False,
    )
    return str(tgt)


def _load_llama(
    path: str, threads: int, *, embed: bool = False, ngl: int | None = None
):
    from llama_cpp import Llama

    return Llama(
        model_path=path,
        n_ctx=default_ctx(),
        n_threads=threads,
        n_gpu_layers=dynamic_gpu_layers(ngl or 32),
        logits_all=False,
        embedding=embed,
    )


def _llama_embed(llm, text: str) -> np.ndarray:
    return np.array(llm.embed(text), dtype="float32")


def _hits(
    qa: List[Dict],
    passages: List[str],
    *,
    knn: Callable[[np.ndarray, int], List[int]],
    q_embed: Callable[[str], np.ndarray],
    k: int,
) -> float:
    good = 0
    for row in qa:
        idx = knn(q_embed(row["question"]), k=k)
        if any(row["answer"].lower() in passages[i].lower() for i in idx):
            good += 1
    return good / len(qa)


def _uplift(a: float, b: float) -> tuple[float, float]:
    delta = b - a
    return delta, (delta / a * 100) if a else float("inf")


def judge_answers_openai(
    items: List[Dict],
    model: str = "gpt-4o-mini",
) -> List[int]:
    """Use an OpenAI model to grade answers – 1 == match, 0 == mismatch."""
    from openai import OpenAI

    client = OpenAI()
    results: List[int] = []

    for it in tqdm(items, desc="LLM-judge"):
        prompt = build_judge_user(it["question"], it["reference"], it["candidate"])
        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_JUDGE},
                        {"role": "user", "content": prompt},
                    ],
                )
                token = resp.choices[0].message.content.strip()
                results.append(1 if token.startswith("1") else 0)
                break
            except Exception as e:
                logging.warning("Judge API error %s – retrying in 5 s", e)
                time.sleep(5)

    return results


def evaluate(cfg_path: str = "config.yaml"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]
    ev = cfg["evaluate"]

    chunks = _load_jsonl(paths["pdf_chunks_jsonl"])
    passages = [c["text"] for c in chunks]

    eval_set = (
        _load_jsonl(paths["eval_manual_jsonl"])
        + _load_jsonl(paths["eval_holdout_jsonl"])
    )[: ev.get("max_eval", 9999)]
    logging.info("Eval set size: %d", len(eval_set))

    retriever = Retriever(device="cuda" if cuda.is_available() else "cpu").fit(
        chunks
    )

    base_path = _ensure_base_gguf(paths["base_gguf"])
    llama_base = _load_llama(base_path, ev["n_threads"], embed=True)
    llama_tune = _load_llama(paths["finetuned_gguf"], ev["n_threads"], embed=True)

    hits_base = _hits(
        eval_set,
        passages,
        knn=retriever._knn,
        q_embed=lambda q: _llama_embed(llama_base, q),
        k=ev["top_k"],
    )
    hits_tune = _hits(
        eval_set,
        passages,
        knn=retriever._knn,
        q_embed=lambda q: _llama_embed(llama_tune, q),
        k=ev["top_k"],
    )

    judge_items_b, judge_items_t = [], []
    for row in tqdm(eval_set, desc="generate answers"):
        q = row["question"]
        context = retriever(q, n_final=ev["top_k"])
        prompt = build_rag_prompt(context, q)

        ans_b = llama_base(prompt, max_tokens=128, temperature=0.2)["choices"][0][
            "text"
        ].strip()
        ans_t = llama_tune(prompt, max_tokens=128, temperature=0.2)["choices"][0][
            "text"
        ].strip()

        judge_items_b.append(
            {"question": q, "reference": row["answer"], "candidate": ans_b}
        )
        judge_items_t.append(
            {"question": q, "reference": row["answer"], "candidate": ans_t}
        )

    acc_base = float(np.mean(judge_answers_openai(judge_items_b)))
    acc_tune = float(np.mean(judge_answers_openai(judge_items_t)))

    d_hits_pp, d_hits_pct = _uplift(hits_base, hits_tune)
    d_acc_pp, d_acc_pct = _uplift(acc_base, acc_tune)

    print("\n=== Evaluation Summary ===")
    print(f"{'Metric':<15}{'Base':>7}{'Tuned':>8}{'Δ(pp)':>9}{'Δ(%)':>9}")
    print("-" * 50)
    print(
        f"Hits@{ev['top_k']:<10}{hits_base:7.2f}{hits_tune:8.2f}"
        f"{d_hits_pp:+9.2f}{d_hits_pct:+8.0f}%"
    )
    print(
        f"LLM-judge     {acc_base:7.2f}{acc_tune:8.2f}"
        f"{d_acc_pp:+9.2f}{d_acc_pct:+8.0f}%\n"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    argp = argparse.ArgumentParser()
    argp.add_argument("--config", default="config.yaml")
    evaluate(argp.parse_args().config)
