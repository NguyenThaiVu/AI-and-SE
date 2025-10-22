#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Continue-pretrain T5 on Python code using span-corruption (T5 denoising).
- Uses your custom SentencePiece tokenizer directory (with extra_ids).
- Uses Hugging Face's DataCollatorForT5MLM to create masked spans on the fly.

Usage:
  python pretrain_t5_span_corruption.py \
      --tokenizer_dir tokenizer_t5_code \
      --output_dir t5_code_pretrained \
      --base_model t5-small

Notes:
- Put your Python functions in the FUNCTIONS list below, or pass text files via --input_files.
- Start with t5-small to validate the pipeline; scale up once it works.
"""

import argparse
import io
import os
from typing import List, Optional
import pandas as pd
import numpy as np

import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    T5Tokenizer,  # or T5TokenizerFast if you prefer
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    set_seed,
)

from t5_span_collator import DataCollatorForT5SpanCorruption

# ------------------- Read dataset -------------------------------

PATH_FILE_DATA = os.path.join(os.getcwd(), "dataset", "processed", "processed_data.csv")

df = pd.read_csv(PATH_FILE_DATA)
list_python_function = df["method_code"].tolist()

FUNCTIONS = list_python_function[:500_000]  # Limit for testing

# ---------------------------------------------------------------------


def read_text_files(paths: List[str]) -> List[str]:
    corpus = []
    for p in paths:
        with io.open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    corpus.append(line)
    return corpus


def make_dataset(functions: List[str]) -> Dataset:
    # One function per row under "text"
    return Dataset.from_dict({"text": functions})


def tokenize_fn(tokenizer, max_len):
    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    return _tok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer_dir",
        type=str,
        required=True,
        help="Path to your saved T5 tokenizer (with spm model, special tokens)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the continued-pretrained model+tokenizer",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="t5-small",
        help="Which T5 checkpoint to continue-pretrain (t5-small/base/large, etc.)",
    )
    parser.add_argument("--input_length", type=int, default=512)
    parser.add_argument("--target_length", type=int, default=128)
    parser.add_argument("--noise_density", type=float, default=0.15)
    parser.add_argument("--mean_span_length", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)  # TODO: increase 
    parser.add_argument("--lr", type=float, default=1e-3)  # good with Adafactor for T5
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # -------- Load tokenizer (your custom SentencePiece + extra_ids) --------
    # Prefer slow tokenizer for simplicity/robustness with SPM; fast also works.
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_dir, use_fast=False)

    # Safety: make sure special IDs are set
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    # -------- Load base T5 and resize embeddings to tokenizer size --------
    model = T5ForConditionalGeneration.from_pretrained(args.base_model)
    model.resize_token_embeddings(len(tokenizer))  # important when vocab size differs

    # Ensure generation IDs are present (useful later during fine-tuning/inference)
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id

    # -------- Build dataset from list and/or files --------
    funcs = [s.strip() for s in FUNCTIONS if s and s.strip()]

    ds = make_dataset(funcs)

    # Tokenize ONLY to get input_ids; collator will create masked inputs/labels
    tokenized = ds.map(
        tokenize_fn(tokenizer, max_len=args.input_length),
        batched=True,
        remove_columns=["text"],
    )

    # -------- Collator that performs span corruption on-the-fly --------
    collator = DataCollatorForT5SpanCorruption(
        tokenizer=tokenizer,
        noise_density=args.noise_density,  # e.g., 0.15
        mean_span_length=args.mean_span_length,  # e.g., 3.0
        input_length=args.input_length,  # e.g., 512
        target_length=args.target_length,  # e.g., 128
        seed=args.seed,
    )
    # -------- Training setup --------
    # Adafactor is lightweight and standard for T5
    # Use fp16 (or bf16) if your GPU supports it
    use_bf16 = (
        torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    )  # Ampere+
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1_000,
        save_steps=1_000,
        save_total_limit=3,
        report_to="none",
        fp16=(torch.cuda.is_available() and not use_bf16),
        bf16=use_bf16,
        optim="adafactor",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    # -------- Train --------
    trainer.train()

    # -------- Save (model + tokenizer together) --------
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n[OK] Pretraining finished.")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
