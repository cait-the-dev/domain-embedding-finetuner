from __future__ import annotations

import argparse
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List

import yaml
from llama_cpp import Llama

from src.prompts import build_rag_prompt
from src.retrieval import Retriever
from src.utils import default_ctx, num_tokens


def load_jsonl(path: str | Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh]


def main(cfg_path: str = "config.yaml", device: str = "cuda"):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    paths = cfg["paths"]
    demo_cfg = cfg.get("demo", {})
    top_k = demo_cfg.get("top_k", 3)
    n_threads = demo_cfg.get("n_threads", 8)

    retriever = Retriever(device=device).fit(load_jsonl(paths["pdf_chunks_jsonl"]))

    llm = Llama(
        model_path=paths["finetuned_gguf"],
        n_threads=n_threads,
        n_gpu_layers=32,
        n_ctx=default_ctx(),
        logits_all=False,
        verbose=False,
    )

    print("\nRAG demo ready. Type a question (blank line to quit):\n")
    try:
        while True:
            question = input("> ").strip()
            if not question:
                break

            t0 = time.time()
            ctx_budget = default_ctx() - 128
            context = retriever(
                question,
                k=top_k * 8,
                max_ctx_tokens=ctx_budget,
            )
            while True:
                prompt = build_rag_prompt(context, question)
                if num_tokens(prompt) <= ctx_budget:
                    break
                context = context[:-1]

            answer = llm(
                prompt,
                max_tokens=128,
                temperature=0.2,
                repeat_penalty=1.1,
                stop=["###"],
                echo=False,
            )["choices"][0]["text"].strip()

            m = re.search(r"\b(?:list|name|give)\s+(\d+)\b", question, re.I)
            if m:
                want = int(m.group(1))
                got = len(re.findall(r"\[[0-9]+\]", answer))
                if got != want:
                    follow = (
                        prompt
                        + f"\n\nThe previous answer listed {got} items but exactly {want} are required. "
                        f"Please list **exactly {want} items** with citations."
                    )
                    answer = llm(follow, max_tokens=128, temperature=0.2)["choices"][
                        0
                    ]["text"].strip()

            print(f"\n{answer}\n--- ({(time.time() - t0)*1000:.0f} ms)\n")
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    args = parser.parse_args()
    main(cfg_path=args.config, device=args.device)
