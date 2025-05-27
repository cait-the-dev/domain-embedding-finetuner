import json
import pathlib
import random

TRAIN = pathlib.Path("data/train.jsonl")
OUT = pathlib.Path("data/eval_holdout.jsonl")
N = 30

rows = [json.loads(l) for l in TRAIN.read_text().splitlines()]
random.seed(42)
sample = random.sample(rows, k=N)

OUT.write_text("\n".join(json.dumps(r) for r in sample) + "\n")
print(f"Wrote {N} held-out pairs to {OUT}")
