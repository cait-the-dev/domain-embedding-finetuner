import argparse
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

from utils import load_config, load_jsonl


class QADataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len=2048):
        self.tok = tok
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        qa = self.rows[idx]
        prompt = f"### Question:\n{qa['question']}\n\n### Answer:\n{qa['answer']}\n"
        ids = self.tok(prompt, truncation=True, max_length=self.max_len, return_tensors="pt")
        ids["labels"] = ids["input_ids"].clone()
        return ids


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    train_rows = load_jsonl(cfg["paths"]["train_jsonl"])

    base_model = cfg["model"]["name"]
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

    model = AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_cfg, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"].get("dropout", 0.1),
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    ds = QADataset(train_rows, tokenizer)

    out_dir = Path(cfg["paths"]["ft_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch"],
        gradient_accumulation_steps=cfg["train"].get("grad_accum", 4),
        optim="adamw_torch",
        learning_rate=cfg["train"].get("lr", 2e-4),
        fp16=False,
        logging_steps=10,
        save_strategy="no",
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    # merge and convert to gguf
    model.save_pretrained(out_dir / "peft")
    print("[finetune] PEFT adapter saved. To merge & convert ➜ convert_to_gguf.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    main(**vars(parser.parse_args()))