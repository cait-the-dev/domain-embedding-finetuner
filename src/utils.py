from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import yaml
from torch import cuda

ROOT = Path(__file__).resolve().parent.parent
_USE_HF: bool = bool(int(os.getenv("USE_HF_TOKENIZER", "0")))


def set_use_hf_tokenizer(flag: bool):
    global _USE_HF, _num_tokens
    _USE_HF = flag
    _num_tokens = _build_token_counter()


def _build_token_counter() -> Callable[[str], int]:
    if _USE_HF:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained("bert-base-uncased")
            return lambda s: len(tok.encode(s))
        except Exception as e:
            print("[utils] HF tokenizer unavailable –", e)

    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return lambda s: len(enc.encode(s))
    except Exception:
        return lambda s: len(re.findall(r"\w+", s))


_num_tokens: Callable[[str], int] = _build_token_counter()


def num_tokens(text: str) -> int:
    return _num_tokens(text)


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    with Path(path or ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_jsonl(data: List[Dict[str, Any]], path: str | Path, mode: str = "w"):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_text(text: str, max_tokens: int, overlap: float = 0.25) -> List[str]:
    tokens = text.split()
    stride = int(max_tokens * (1 - overlap)) or max_tokens
    min_len = int(max_tokens * 0.8)

    chunks: List[str] = []
    idx = 0
    while idx < len(tokens):
        window_tokens = tokens[idx : idx + max_tokens]
        if len(window_tokens) < min_len:
            break
        chunk = " ".join(window_tokens)
        if _num_tokens(chunk) >= 40:
            chunks.append(chunk)
        idx += stride
    return chunks


def dynamic_gpu_layers(requested: int = 32) -> int:
    return requested if cuda.is_available() else 0


def default_ctx(model_ctx: int | None = None) -> int:
    return int(
        os.getenv(
            "LLAMA_MAX_CTX", 4096 if model_ctx is None else min(model_ctx, 8192)
        )
    )
