from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from huggingface_hub import snapshot_download


def pull_repo(
    repo_id: str = "meta-llama/Llama-3.2-1B",
    out_dir: str | Path = "models/llama-3.2-1b",
    revision: str | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"→ downloading {repo_id} to {out_dir} …")
    snapshot_download(
        repo_id=repo_id,
        revision=revision or "main",
        local_dir=str(out_dir),
        token=os.getenv("HUGGINGFACE_HUB_TOKEN"),
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8,
    )
    print("✓ download complete")
    return out_dir


def to_gguf(
    model_dir: Path,
    quant: str = "Q4_K_M",
    llama_cpp_root: str | Path = "llama.cpp",
) -> None:
    ckpt = next(model_dir.glob("*.bin"), None)
    if ckpt is None:
        raise FileNotFoundError("no *.bin weights found in model directory")

    out_path = model_dir.with_suffix(f".{quant}.gguf")

    print(f"→ converting to GGUF ({quant}) …")
    cmd = [
        "./convert.py",
        "--outtype",
        "gguf",
        "--outfile",
        str(out_path),
        "--quantize",
        quant,
        str(ckpt),
    ]
    subprocess.run(cmd, cwd=llama_cpp_root, check=True)
    print(f"✓ GGUF written → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="models/llama-3.2-1b")
    ap.add_argument("--revision", help="git sha / tag / branch")
    ap.add_argument("--gguf", metavar="TYPE", help="quant type, e.g. Q4_K_M")
    ap.add_argument("--llama_cpp_root", default="llama.cpp")
    args = ap.parse_args()

    if not os.getenv("HUGGINGFACE_HUB_TOKEN"):
        raise SystemExit("set HUGGINGFACE_HUB_TOKEN first")

    model_dir = pull_repo(
        out_dir=args.out_dir,
        revision=args.revision,
    )

    if args.gguf:
        to_gguf(model_dir, quant=args.gguf, llama_cpp_root=args.llama_cpp_root)


if __name__ == "__main__":
    main()
