from __future__ import annotations

import argparse, json, logging, time
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
import yaml
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def load_config(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

def load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def cosine(u: np.ndarray, v: np.ndarray):
    return 1 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)


def build_faiss(chunks: List[Dict], embed_model: SentenceTransformer):
    texts = [c["text"] for c in chunks]
    embs = embed_model.encode(texts, show_progress_bar=True, normalize_embeddings=True).astype(
        "float32"
    )
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    return index, embs


def retrieve_hits(
    qa_set: List[Dict],
    chunks: List[Dict],
    embed_model: SentenceTransformer,
    index: faiss.Index,
    k: int = 3,
):
    chunk_texts = [c["text"] for c in chunks]
    hits = 0
    for row in qa_set:
        q_emb = embed_model.encode(row["question"], normalize_embeddings=True)
        D, I = index.search(np.array([q_emb], dtype="float32"), k)
        top_texts = [chunk_texts[i] for i in I[0]]
        if any(row["answer"].strip().lower() in t.lower() for t in top_texts):
            hits += 1
    return hits / len(qa_set)


SYSTEM_JUDGE = """You are an impartial grader. 
Given a question, the reference answer, and a candidate answer, 
return "1" if the candidate answer means the same thing as the reference answer
(even if wording differs) and return "0" otherwise.
Return ONLY "1" or "0" with no other text."""

def judge_answers_openai(items: List[Dict], model="gpt-4o-mini"):
    """items: [{'question','reference','candidate'}] -> list[int]"""
    from openai import OpenAI
    client = OpenAI()
    results = []
    for it in tqdm(items, desc="LLM-judge"):
        prompt = (
            f"Question: {it['question']}\n"
            f"Reference answer: {it['reference']}\n"
            f"Candidate answer: {it['candidate']}\n"
            "Does the candidate match the reference? Return 1 or 0."
        )
        while True:
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_JUDGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                )
                token = resp.choices[0].message.content.strip()
                results.append(1 if token.startswith("1") else 0)
                break
            except Exception as e:
                logging.warning("Judge API error %s – retrying in 5 s", e)
                time.sleep(5)
    return results


def load_llama(model_path: str, n_threads: int = 8):
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_threads=n_threads, n_gpu_layers=0, logits_all=False)

def answer_with_llama(llm, prompt: str, max_tokens: int = 128) -> str:
    out = llm(
        prompt,
        max_tokens=max_tokens,
        echo=False,
        temperature=0.2,
        repeat_penalty=1.1,
    )
    return out["choices"][0]["text"].strip()


def evaluate(cfg_path="config.yaml"):
    cfg = load_config(cfg_path)
    paths = cfg["paths"]
    ev_cfg = cfg["evaluate"]

    chunks = load_jsonl(paths["pdf_chunks_jsonl"])
    eval_set = load_jsonl(paths["eval_manual_jsonl"]) + load_jsonl(paths["eval_holdout_jsonl"])
    eval_set = eval_set[: ev_cfg.get("max_eval", len(eval_set))] 

    logging.info("Eval set size: %d", len(eval_set))

    logging.info("Loading sentence-transformer …")
    st_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    index, _ = build_faiss(chunks, st_model)

    hits_base = retrieve_hits(eval_set, chunks, st_model, index, k=ev_cfg["top_k"])
    logging.info("Hits@%d (embedding retrieval): %.3f", ev_cfg["top_k"], hits_base)

    logging.info("Loading base + fine-tuned gguf …")
    llama_base = load_llama(cfg["finetune"]["base_model_path_gguf"], ev_cfg["n_threads"])
    llama_tuned = load_llama(paths["finetuned_gguf"], ev_cfg["n_threads"])

    judge_items_base, judge_items_tuned = [], []

    for row in tqdm(eval_set, desc="gen answers"):
        q = row["question"]
        q_emb = st_model.encode(q, normalize_embeddings=True)
        _, I = index.search(np.array([q_emb], dtype="float32"), ev_cfg["top_k"])
        context = "\n".join(chunks[i]["text"] for i in I[0])
        prompt = f"Context:\n{context}\n\nQuestion: {q}\nAnswer:"

        ans_base = answer_with_llama(llama_base, prompt)
        ans_tune = answer_with_llama(llama_tuned, prompt)

        judge_items_base.append(
            {"question": q, "reference": row["answer"], "candidate": ans_base}
        )
        judge_items_tuned.append(
            {"question": q, "reference": row["answer"], "candidate": ans_tune}
        )

    acc_base = np.mean(judge_answers_openai(judge_items_base))
    acc_tune = np.mean(judge_answers_openai(judge_items_tuned))

    def uplift(b, t):
        delta_pp = t - b
        delta_pct = (delta_pp / b * 100) if b > 0 else float("inf")
        return delta_pp, delta_pct

    dpp_hits, dpct_hits = uplift(hits_base, hits_base) 
    dpp_ans, dpct_ans = uplift(acc_base, acc_tune)

    print("\n=== Evaluation Summary ===\n")
    print(f"{'Metric':<15} {'Base':>6} {'Tuned':>6} {'Δ(pp)':>8} {'Δ(%)':>8}")
    print("-" * 45)
    print(
        f"{'Hits@'+str(ev_cfg['top_k']):<15} "
        f"{hits_base:6.2f} {hits_base:6.2f} {dpp_hits:+8.2f} {dpct_hits:+7.0f}%"
    )
    print(
        f"{'LLM-judge':<15} "
        f"{acc_base:6.2f} {acc_tune:6.2f} {dpp_ans:+8.2f} {dpct_ans:+7.0f}%"
    )
    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    evaluate(args.config)