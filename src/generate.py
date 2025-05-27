from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.prompts import SYSTEM_QA_GEN, USER_QA_BATCH
from src.utils import load_config, load_jsonl, save_jsonl

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    from openai import OpenAI
except ImportError as e:
    raise SystemExit("openai package required: `pip install openai`") from e


DEFAULT_BATCH = 25
MAX_ROWS = 500


def _create_batches(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _qa_from_response(line: str) -> Optional[Dict[str, str]]:
    try:
        qa = json.loads(line)
    except json.JSONDecodeError:
        m = re.match(r"Q:\s*(.+?)\n\s*A:\s*(.+)", line, flags=re.I | re.S)
        if not m:
            return None
        qa = {"question": m.group(1).strip(), "answer": m.group(2).strip()}

    ans_len = len(qa.get("answer", ""))
    if 10 <= ans_len <= 200 and qa.get("question"):
        return {"question": qa["question"].strip(), "answer": qa["answer"].strip()}
    return None


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def _openai_chat(client: OpenAI, messages):
    return client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=512,
        messages=messages,
    )


def generate_qa_pairs(
    chunks: List[Dict[str, str]],
    *,
    batch_size: int = DEFAULT_BATCH,
    max_rows: int = MAX_ROWS,
) -> List[Dict[str, str]]:
    passages = [c["text"] if isinstance(c, dict) else c for c in chunks]
    random.shuffle(passages)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set; export or add to .env")
    client = OpenAI(api_key=api_key)

    qa_pairs: List[Dict[str, str]] = []
    for batch in _create_batches(passages, batch_size):
        if len(qa_pairs) >= max_rows:
            break

        prompt_msgs = [
            {"role": "system", "content": SYSTEM_QA_GEN},
            {
                "role": "user",
                "content": USER_QA_BATCH.format(passages="\n---\n".join(batch)),
            },
        ]

        try:
            resp = _openai_chat(client, prompt_msgs)
        except Exception as err:
            print("[generate] OpenAI error →", err)
            continue

        for raw in resp.choices[0].message.content.strip().splitlines():
            qa = _qa_from_response(raw)
            if qa:
                qa_pairs.append(qa)
            if len(qa_pairs) >= max_rows:
                break

        print(f"[generate] total QA pairs: {len(qa_pairs)}")
        time.sleep(0.5)

    return qa_pairs[:max_rows]


def main(config: str = "config.yaml"):
    cfg = load_config(config)
    paths = cfg["paths"]
    gen_cfg = cfg.get("generate", {})

    chunks = load_jsonl(paths["pdf_chunks_jsonl"])
    batch_size = gen_cfg.get("batch_size", DEFAULT_BATCH)
    max_rows = min(gen_cfg.get("max_rows", MAX_ROWS), MAX_ROWS)

    qa_pairs = generate_qa_pairs(chunks, batch_size=batch_size, max_rows=max_rows)

    out_path = Path(paths["train_jsonl"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(qa_pairs, out_path)
    print(f"[generate] wrote {len(qa_pairs)} pairs → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(**vars(parser.parse_args()))
