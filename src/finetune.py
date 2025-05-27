import argparse
import logging
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from src.utils import load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


def make_dataset(train_path: str, tokenizer, max_len: int):
    raw_ds = load_dataset("json", data_files=train_path, split="train")

    def _tokenize(example):
        prompt = f"<s>[INST] {example['question']} [/INST] {example['answer']} </s>"
        ids = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )["input_ids"]
        return {"input_ids": ids, "labels": ids.copy()}

    tokenized = raw_ds.map(_tokenize, remove_columns=list(raw_ds.features))
    return tokenized


def _model_available(id_or_path: str) -> bool:
    p = Path(id_or_path)
    return p.is_dir() and any(p.glob("*.bin")) and (p / "tokenizer.json").exists()


def main(cfg_path: str):
    cfg = load_config(cfg_path)
    ft_cfg = cfg["finetune"]
    paths = cfg["paths"]

    base_id = ft_cfg["base_model_id"]

    if not _model_available(base_id):
        logging.warning(
            "[finetune] Full base model not found at %s – skipping finetune step.",
            base_id,
        )
        return

    tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    logging.info("Loading base model …")
    base_model = AutoModelForCausalLM.from_pretrained(
        ft_cfg["base_model_id"], quantization_config=bnb_cfg, device_map="auto"
    )
    base_model = prepare_model_for_kbit_training(base_model)

    lora = LoraConfig(
        r=ft_cfg["lora_r"],
        lora_alpha=ft_cfg["lora_alpha"],
        lora_dropout=ft_cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ft_cfg["target_modules"],
    )
    model = get_peft_model(base_model, lora)
    model.print_trainable_parameters()

    train_jsonl = paths["train_jsonl"]
    ds = make_dataset(train_jsonl, tokenizer, ft_cfg["max_seq_len"])

    training_args = TrainingArguments(
        output_dir=ft_cfg["output_dir"],
        num_train_epochs=ft_cfg["num_epochs"],
        per_device_train_batch_size=ft_cfg["batch_size"],
        gradient_accumulation_steps=ft_cfg["grad_accum"],
        learning_rate=ft_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=lambda data: {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f["input_ids"]) for f in data],
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "labels": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(f["labels"]) for f in data],
                batch_first=True,
                padding_value=-100,
            ),
        },
    )

    logging.info("Start training …")
    trainer.train()
    logging.info("Training done.")

    model.save_pretrained(ft_cfg["output_dir"])
    tokenizer.save_pretrained(ft_cfg["output_dir"])

    logging.info("Merging LoRA adapter into base weights …")
    merged = model.merge_and_unload()
    merged.save_pretrained(ft_cfg["output_dir"] + "/merged")

    gguf_out = Path(ft_cfg["merged_gguf"]).expanduser()
    logging.info("Converting to GGUF → %s", gguf_out)
    try:
        from llama_cpp import convert
    except ImportError:
        logging.error("llama_cpp not installed; skip GGUF conversion.")
    else:
        convert(
            model_path=(ft_cfg["output_dir"] + "/merged"),
            out_path=str(gguf_out),
            vocab_type="llama",
            use_mmap=False,
        )
    logging.info("Finetune complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
