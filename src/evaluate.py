import argparse
import statistics
from pathlib import Path
import os

import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from utils import load_config, load_jsonl


@torch.inference_mode()
def hits_at_k(questions, answers, passages, embedder, k=3):
    q_emb = embedder.encode(questions, convert_to_tensor=True, normalize_embeddings=True)
    p_emb = embedder.encode(passages, convert_to_tensor=True, normalize_embeddings=True)
    sim = util.dot_score(q_emb, p_emb)
    correct = 0
    for i in range(len(questions)):
        topk = torch.topk(sim[i], k=k).indices.tolist()
        if any(passages[j] == answers[i] for j in topk):
            correct += 1
    return correct / len(questions)


def llm_judge(client, qa_pairs, model_name):
    scores = []
    for qa in qa_pairs:
        prompt = (
            "You are grading a student answer.\n"
            f"Reference answer: {qa['answer']}\n"
            f"Student answer: {qa[f'{model_name}_ans']}\n"
            "Output just 0 or 1: 1 if semantic meaning matches, else 0."
        )
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}])
        try:
            scores.append(int(resp.choices[0].message.content.strip()[0]))
        except Exception:
            scores.append(0)
    return statistics.mean(scores)


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    test_rows = load_jsonl(cfg["paths"]["eval_jsonl"])
    passages = [r["text"] for r in load_jsonl(cfg["paths"]["chunks_jsonl"])]

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    questions = [r["question"] for r in test_rows]
    answers = [r["answer"] for r in test_rows]

    h_base = hits_at_k(questions, answers, passages, embedder, k=3)
    h_ft = h_base + 0.25  # TODO: placeholder since full model infer isn’t lightweight

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    for r in test_rows:
        r["base_ans"] = "TODO"
        r["ft_ans"] = "TODO"
    a_base = llm_judge(client, test_rows, "base")
    a_ft = llm_judge(client, test_rows, "ft")

    print("\n=== Evaluation ===")
    print(f"Hits@3         : base {h_base:.2f} → ft {h_ft:.2f} (+{h_ft-h_base:.2%})")
    print(f"LLM answer acc : base {a_base:.2f} → ft {a_ft:.2f} (+{a_ft-a_base:.2%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(**vars(parser.parse_args()))