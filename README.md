# domain-embedding-finetuner  
**Lightweight pipeline to parse a domain-specific PDF, generate training data, fine-tune a Llama-3.2 (1B) SLM, and compare embedding performance against the base model.**

---

## 1 . Overview
This repo contains a **fully-offline, end-to-end pipeline** that

1. ingests a single PDF,  
2. creates ≤ 500 synthetic Q-A pairs with GPT-4o-mini,  
3. parameter-efficiently fine-tunes a 1 B-parameter Llama-3 model (4-bit QLoRA + LoRA),  
4. quantizes the result to **GGUF** for ultra-fast local inference via **llama.cpp**,  
5. evaluates the fine-tuned model vs. the base checkpoint on a held-out test set,  
6. (optionally) launches a CLI demo that answers questions over the document.

All code is < 400 LOC, modular, and reproducible with a single command:

```bash
make all          # ingest ▸ generate_QA ▸ finetune ▸ evaluate
```

## 2 . Directory Layout
``` graphql
.
├── data/
│   ├── manual.pdf
│   └── qa_pairs.jsonl         # auto-generated
├── models/
│   ├── base/                  # downloaded Llama-3.2-1B model
│   └── finetuned.gguf         # final 4-bit model (after make all)
├── src/
│   ├── ingest.py              # ① PDF ➜ cleaned chunks
│   ├── generate.py            # ② GPT-4o QA generation
│   ├── finetune.py            # ③ LoRA + QLoRA training
│   ├── evaluate.py            # ④ metrics & report
│   └── demo.py                # ⑤ (bonus) retrieval + chat CLI
├── config.yaml                # single source of truth for paths & params
├── requirements.txt
├── Dockerfile                 # CPU and CUDA variants (offline-friendly)
└── README.md
```
## 3 . Quick-Start
### 3.1 Prerequisites
Component	Min Version	Notes
Python	3.10	venv recommended
GCC/Clang	11+	for llama.cpp build
Optional GPU	16 GB VRAM	cuts training time to ~30 min

Offline use: download the base model (`QuantFactory/Llama-3.2-1B-GGUF`) and place it in models/base/ before running make all.

### 3.2 Install & Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all
```
The pipeline will:

1. extract & chunk the PDF (`data/pdf_chunks.jsonl`),
2. contact GPT-4o-mini (API key must be set as `OPENAI_API_KEY`) to build `qa_pairs.jsonl`,
3. run QLoRA training and write `models/finetuned.gguf`,
4. print an evaluation table like:

Metric	Base	Fine-Tuned
Hits@3	0.56	0.81
LLM-Judge Accuracy	47 %	73 %

### 3.3 Run Demo (bonus)
``` bash
make demo          # spins up llama.cpp and opens an interactive prompt
```
Ask questions such as:

``` sql
> What are the three phases of troop leading procedures?
```

## 4 . Configuration
All knobs live in `config.yaml`

```yaml
model:
  base_path:   models/base
  lora_r:      8
  lora_alpha:  32
  load_4bit:   true
data:
  pdf_path:    data/manual.pdf
  chunk_tokens: 300
  overlap:     0.25
training:
  epochs:      2
  lr:          2e-4
  batch_size:  8
evaluation:
  manual_questions: data/manual_eval.json
```
Change paths/hyper-params once; every script reads the same file.

## 5 . Implementation Notes
Ingest uses `pdfplumber`, regex clean-ups, and token-based sliding window.

QA Generation batches 25 chunks / request to GPT-4o-mini; temp = 0.2 to keep deterministic.

Fine-Tune leverages HuggingFace `peft` with 4-bit NF4 quantization (`bitsandbytes`).

Export adapter weights are merged and converted to GGUF → compatible with `llama.cpp` CPU runtime.

Evaluation implements simple cosine-similarity retrieval + GPT-4o judge prompt; metrics saved to `results/`.

Offline Path If GPT-4o is not allowed, swap `generate.py` prompt engine to an on-prem teacher – only that one line changes.
