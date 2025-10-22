import re
import random
from typing import List, Dict, Tuple, Optional
import os 
import math
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset
from datasets import Dataset, DatasetDict
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    set_seed,
)

PREFIX   = "predict_if_condition: "
SENTINEL = "<extra_id_0>"


def _mask_span_by_pos(code: str, start_line: int, start_col: int, end_line: int, end_col: int, sentinel=SENTINEL) -> str:
    """Replace the exact character span [test] with the sentinel, preserving everything else."""
    lines = code.splitlines(keepends=True)
    before = "".join(lines[:start_line-1]) + lines[start_line-1][:start_col]
    after  = lines[end_line-1][end_col:] + "".join(lines[end_line:])
    return before + sentinel + after


# def extract_and_mask_first_if(code: str) -> Optional[Tuple[str, str]]:
#     """
#     Returns (masked_code, condition_text) for the first 'if' in the first def.
#     Tries AST first (preferred), falls back to a regex if AST unparsing fails.
#     """
#     code = code.strip()
#     if not code:
#         return None

#     # Prefer AST (handles whitespace/parens better)
#     try:
#         import ast
#         tree = ast.parse(code)

#         class Finder(ast.NodeVisitor):
#             def __init__(self):
#                 self.cond_src = None

#             def visit_FunctionDef(self, node: ast.FunctionDef):
#                 if self.cond_src is None:
#                     self.generic_visit(node)

#             def visit_If(self, node: ast.If):
#                 if self.cond_src is None:
#                     try:
#                         self.cond_src = ast.unparse(node.test)
#                     except Exception:
#                         self.cond_src = None
#                 # stop after the first if
#                 return

#         f = Finder()
#         f.visit(tree)
#         if f.cond_src:
#             test_pat = re.escape(f.cond_src)
#             # Replace first "if <cond>:" with sentinel
#             pat = r"(if\s+)" + test_pat + r"(\s*:)"
#             masked, n = re.subn(pat, r"\1" + SENTINEL + r"\2", code, count=1)
#             if n > 0:
#                 return masked, f.cond_src
#     except Exception:
#         pass

#     # Fallback regex: first line that looks like `if ... :`
#     m = re.search(r"(?m)^\s*if\s+(?P<cond>.+?)\s*:\s*$", code)
#     if m:
#         cond = m.group("cond")
#         # Replace only within that match span to avoid accidental other replacements
#         start, end = m.span()
#         line = code[start:end]
#         line_masked = line.replace(cond, SENTINEL, 1)
#         masked = code[:start] + line_masked + code[end:]
#         return masked, cond.strip()

#     return None

def extract_mask_all_ifs(code: str) -> List[Tuple[str, str]]:
    """
    Return (masked_code, cond_text) for EVERY ast.If in EVERY function in 'code'.
    - Uses exact source spans via ast.get_source_segment when possible.
    - Covers `elif` (which are ast.If nodes in orelse), nested ifs, multiple functions.
    """
    out: List[Tuple[str, str]] = []
    code = code.strip("\n")
    if not code:
        return out

    try:
        tree = ast.parse(code)
    except Exception:
        return out

    # collect every ast.If node with positional info
    if_nodes: List[ast.If] = [n for n in ast.walk(tree) if isinstance(n, ast.If)]
    # deterministically order by source position
    def _key(n: ast.If):
        return (getattr(n, "lineno", 10**9), getattr(n, "col_offset", 10**9))
    if_nodes.sort(key=_key)

    for n in if_nodes:
        # need Python 3.8+ for end_lineno/end_col_offset
        if not (hasattr(n.test, "lineno") and hasattr(n.test, "end_lineno")):
            continue

        # extract original condition text
        cond_text = None
        try:
            cond_text = ast.get_source_segment(code, n.test)
        except Exception:
            try:
                cond_text = ast.unparse(n.test)  # 3.9+
            except Exception:
                cond_text = None
        if not cond_text:
            continue
        cond_text = cond_text.strip()
        if not cond_text:
            continue

        # replace exactly the test span with the sentinel
        masked = _mask_span_by_pos(
            code,
            n.test.lineno, n.test.col_offset,
            n.test.end_lineno, n.test.end_col_offset,
            sentinel=SENTINEL,
        )
        out.append((masked, cond_text))
    return out


# def build_pairs_from_list(functions: List[str]) -> List[Dict[str, str]]:
#     """
#     From a list of function codes, build (masked_code, condition) pairs.
#     """
#     pairs = []
#     for fn in functions:
#         res = extract_and_mask_first_if(fn)
#         if not res:
#             continue
#         masked, cond = res
#         cond = cond.strip()
#         if not cond:
#             continue
#         pairs.append({
#             "source": f"{PREFIX}{masked}",
#             "target": cond
#         })
#     return pairs

def build_pairs_from_list(functions: List[str]) -> List[Dict[str, str]]:
    """Build one training pair per `if` (mask one at a time)."""
    pairs: List[Dict[str, str]] = []
    for fn_src in functions:
        for masked, cond in extract_mask_all_ifs(fn_src):
            pairs.append({
                "source": f"{PREFIX}{masked}",
                "target": cond,
            })
    return pairs


def make_tokenize_fn(tokenizer: T5Tokenizer, max_src_len: int, max_tgt_len: int):
    def _fn(batch):
        model_inputs = tokenizer(batch["source"], truncation=True, max_length=max_src_len)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(batch["target"], truncation=True, max_length=max_tgt_len)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _fn


def normalize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\(\s*", "(", s)
    s = re.sub(r"\s*\)", ")", s)
    return s

def make_compute_metrics(tokenizer: T5Tokenizer):
    def _cmp(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):  # HF sometimes returns (sequences, …)
            preds = preds[0]
        pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = labels.copy()
        labels[labels == -100] = tokenizer.pad_token_id
        gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)
        exact = sum(normalize(p) == normalize(g) for p, g in zip(pred_texts, gold_texts))
        return {"exact_match": exact / max(1, len(pred_texts))}
    return _cmp


from torch.utils.data import Subset
from transformers import TrainerCallback

class GenerateEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        max_new_tokens=32,          
        num_beams=2,                 
        batch_size=4,                
        sample_size=128,             # eval on a subset each time
        metric_key="exact_match",
        seed=42
    ):
        self.tok = tokenizer
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.metric_key = metric_key
        self.seed = seed
        self.best = -1.0
        self.best_path = None
        random.seed(seed)

    def _norm(self, s: str) -> str:
        s = s.strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\(\s*", "(", s)
        s = re.sub(r"\s*\)", ")", s)
        if s.endswith(":"): s = s[:-1].rstrip()
        return s

    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        if trainer is None:
            return
        model = trainer.model
        model.eval()

        # pick a subset to evaluate to avoid OOM
        eval_ds = trainer.eval_dataset
        if len(eval_ds) > self.sample_size:
            # stable random subset
            idxs = random.sample(range(len(eval_ds)), self.sample_size)
            eval_ds = Subset(eval_ds, idxs)

        # create a small dataloader
        dl = trainer.get_eval_dataloader(eval_dataset=eval_ds)

        # temporarily disable KV cache if requested (reduces memory during generate)
        prev_use_cache = getattr(model.config, "use_cache", True)

        preds, golds = [], []
        with torch.no_grad():
            for batch in dl:
                # move to device & drop token_type_ids
                batch.pop("token_type_ids", None)
                for k in list(batch.keys()):
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(model.device, non_blocking=True)

                # generate in batch
                out = model.generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                    early_stopping=True,
                    decoder_start_token_id=model.config.decoder_start_token_id,
                    eos_token_id=model.config.eos_token_id,
                    pad_token_id=model.config.pad_token_id,
                )

                # decode predictions
                pred_texts = self.tok.batch_decode(out, skip_special_tokens=True)
                preds.extend(self._norm(p) for p in pred_texts)

                # decode gold (replace -100 with pad)
                labels = batch["labels"].clone()
                labels[labels == -100] = self.tok.pad_token_id
                gold_texts = self.tok.batch_decode(labels, skip_special_tokens=True)
                golds.extend(self._norm(g) for g in gold_texts)


        # restore original cache setting
        model.config.use_cache = prev_use_cache

        exact = float(np.mean([p == g for p, g in zip(preds, golds)])) if preds else 0.0
        trainer.log({self.metric_key: exact})
        print(f"\n[generate-eval] subset_size={len(golds)}  beams={self.num_beams}  "
              f"max_new={self.max_new_tokens}  exact_match={exact:.4f}")

        # save best checkpoint (optional)
        if exact > self.best:
            self.best = exact
            self.best_path = os.path.join(args.output_dir, "best-generate")
            trainer.save_model(self.best_path)
            self.tok.save_pretrained(self.best_path)
            print(f"[generate-eval] New best ({exact:.4f}) saved to {self.best_path}")


def finetune_t5_from_list(
    functions: List[str],
    tokenizer_dir: str,
    model_dir: str,
    output_dir: str = "t5_if_finetuned",
    max_src_len: int = 512,
    max_tgt_len: int = 128,
    batch_size: int = 8,
    grad_accum: int = 4,
    epochs: int = 5,
    lr: float = 5e-5,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Fine-tune T5 on (masked function -> condition) using an in-memory list.
    - tokenizer_dir: your custom T5 tokenizer (with extra_ids, spm model).
    - model_dir: base or code-continued T5 (e.g., t5_code_pretrained).
    """
    set_seed(seed)

    # Load tokenizer & model
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    # Ensure generation IDs
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.config.eos_token_id is None and tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
    if model.config.decoder_start_token_id is None:
        model.config.decoder_start_token_id = model.config.pad_token_id

    # Build pairs from the list, then split
    pairs = build_pairs_from_list(functions)
    if not pairs:
        raise ValueError("No (masked, condition) pairs could be built from the provided functions.")

    random.shuffle(pairs)
    n = len(pairs)
    n_val = max(1, int(val_ratio * n))
    train_pairs = pairs[:-n_val] if n_val < n else pairs
    val_pairs   = pairs[-n_val:] if n_val < n else pairs[:]

    ds = DatasetDict({
        "train": Dataset.from_list(train_pairs),
        "validation": Dataset.from_list(val_pairs),
    })

    # Tokenize
    tok_fn = make_tokenize_fn(tokenizer, max_src_len, max_tgt_len)
    ds_tok = ds.map(tok_fn, batched=True, remove_columns=["source", "target"])

    # Collator
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

    # Training args
    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=2_000,
        eval_steps=2_000,
        save_steps=2_000,
        save_total_limit=3,
        # eval_strategy="steps",
        eval_strategy="no",
        logging_strategy="steps",
        save_strategy="steps",
        metric_for_best_model="exact_match",
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        # eval_dataset=ds_tok["validation"],
        eval_dataset=None,
        data_collator=collator,
        # compute_metrics=make_compute_metrics(tokenizer),
        compute_metrics=None
    )

    # trainer.add_callback(GenerateEvalCallback(
    #     tokenizer=tokenizer,
    #     max_new_tokens=24,   
    #     num_beams=1,         
    #     batch_size=4,        
    #     sample_size=128
    # ))

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("Fine-tuning completed. Model and tokenizer saved to", output_dir)

    return output_dir

if __name__ == "__main__":
    
    PATH_FILE_DATA = os.path.join(os.getcwd(), "dataset", "processed", "processed_data.csv")

    df = pd.read_csv(PATH_FILE_DATA)
    list_python_function = df["method_code"].tolist()
    N = len(list_python_function)
    N_train = math.floor(0.9 * N)
    FUNCTIONS = list_python_function[:N_train]  # Limiting to first 1000 functions for testing
    
    finetuned_dir = finetune_t5_from_list(
        functions=FUNCTIONS,
        tokenizer_dir="tokenizer_t5_code",     # your saved tokenizer
        model_dir="t5_code_pretrained",        # or "t5-small" if skipped pretraining
        output_dir="t5_if_finetuned",
        epochs=5,
        batch_size=16,
        lr=5e-5,
    )
    print("Saved fine-tuned model to:", finetuned_dir)