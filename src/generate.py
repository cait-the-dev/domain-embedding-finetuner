from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import List

from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.utils import load_config, load_jsonl, save_jsonl

try:
    from openai import OpenAI 
except ImportError as e:  
    raise SystemExit("openai package required: pip install openai") from e

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

SYSTEM_PROMPT = (
    "You are a military subject‑matter expert. For **each** passage I provide, "
    "write ONE factual question that can be answered exactly from that passage, "
    "then output the answer verbatim from the passage. Return newline‑separated "
    "JSON objects with keys 'question' and 'answer'."
)
USER_TEMPLATE = """Passages separated by \n---\n\n{passages}\n"""


def batch_chunks(chunks: List[str], size: int):
    for i in range(0, len(chunks), size):
        yield chunks[i : i + size]


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def call_openai(client: OpenAI, messages):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=512,
        messages=messages,
    )


def main(config: str = "config.yaml"):
    cfg = load_config(config)
    chunks = [r["text"] for r in load_jsonl(cfg["paths"]["chunks_jsonl"])]
    out_path = Path(cfg["paths"]["train_jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.shuffle(chunks)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set; export or add to .env")
    client = OpenAI(api_key=api_key)

    batch_size = cfg.get("generate", {}).get("batch", 25)
    target_pairs = min(cfg.get("generate", {}).get("max_rows", 500), 500)

    qa_pairs: List[dict] = []
    for chunk_batch in batch_chunks(chunks, batch_size):
        if len(qa_pairs) >= target_pairs:
            break
        passages = "\n---\n".join(chunk_batch)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(passages=passages)},
        ]
        try:
            resp = call_openai(client, messages)
        except Exception as err:
            print("[generate] OpenAI error →", err)
            continue
        for line in resp.choices[0].message.content.strip().splitlines():
            try:
                qa = json.loads(line)
            except json.JSONDecodeError:
                continue
            ans_len = len(qa.get("answer", ""))
            if 10 <= ans_len <= 200:
                qa_pairs.append(qa)
            if len(qa_pairs) >= target_pairs:
                break
        print(f"[generate] total QA pairs: {len(qa_pairs)}")
        time.sleep(0.5)

    save_jsonl(qa_pairs[:target_pairs], out_path)
    print(f"[generate] wrote {len(qa_pairs[:target_pairs])} pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(**vars(parser.parse_args()))
    