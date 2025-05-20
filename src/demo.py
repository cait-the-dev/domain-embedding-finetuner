import argparse
import subprocess
from pathlib import Path

from utils import load_config


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    model_path = Path(cfg["paths"]["gguf_model"])
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cmd = [
        "llama.cpp/main",  # TODO: adjust if llama.cpp binary in PATH
        "--model",
        str(model_path),
        "--threads",
        str(cfg.get("threads", 8)),
        "--n-gpu-layers",
        "0",
    ]
    print("Launching llama.cpp… (Ctrl+C to exit)")
    subprocess.run(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(**vars(parser.parse_args()))
