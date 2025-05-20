from __future__ import annotations

import json
import os
import platform
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT = Path(__file__).resolve().parent.parent

if platform.system() == "Windows":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    p = Path(path or ROOT / "config.yaml")
    with p.open() as f:
        return yaml.safe_load(f)


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def save_jsonl(data: List[Dict[str, Any]], path: str | Path, mode: str = "w"):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


try:
    import tiktoken 

    _enc = tiktoken.get_encoding("cl100k_base")

    def _num_tokens(text: str) -> int:
        return len(_enc.encode(text))
except Exception: 

    def _num_tokens(text: str) -> int:  
        return len(re.findall(r"\w+", text))


def chunk_text(text: str, max_tokens: int, overlap: float = 0.25) -> List[str]:
    tokens = text.split() 
    stride = int(max_tokens * (1 - overlap))
    chunks: List[str] = []
    i = 0
    while i < len(tokens):
        window = tokens[i : i + max_tokens]
        joined = " ".join(window)
        if _num_tokens(joined) >= 40:
            chunks.append(joined)
        i += stride if stride else max_tokens 
    return chunks
