import argparse
from uuid import uuid4

import pdfplumber

from src.utils import chunk_text, load_config, save_jsonl


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    pdf_path = cfg["paths"]["pdf"]
    out_path = cfg["paths"]["pdf_chunks_jsonl"]
    max_tokens = cfg["ingest"]["max_tokens"]
    overlap = cfg["ingest"].get("overlap", 0.25)

    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            for chunk in chunk_text(text, max_tokens, overlap):
                rows.append(
                    {
                        "chunk_id": str(uuid4()),
                        "text": chunk.strip(),
                        "source_page": page_num,
                    }
                )
    save_jsonl(rows, out_path)
    print(f"[ingest] Wrote {len(rows)} chunks ➜ {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="cfg_path", default="config.yaml")
    main(**vars(parser.parse_args()))
