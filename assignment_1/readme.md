# Project: Pre-training and Fine-tuning Transformer Models for Python Code 

Here’s a concise, reader-friendly description you can drop into your README under “Data Crawling”.

---

## 1. Data Crawling & Preparation

**Goal:** build a large, clean corpus of real-world Python functions for model training.

### Process

* Query the 500 GitHub repos in Python with minimum 50 stars.
* For each repo, fetch the Git tree and collect file ending with `.py`. This stage runs with a multi-thread to hide network latency.
* Walk each `.py` file’s using library **AST** and extract python function. This stage uses a **ProcessPool** for parallelism across CPU cores.
* Save a dataset into pandas dataFrame with, deduplicate samples, and write a final CSV `python_methods_raw_final.csv`.

**Note:**

* Network and parse errors are caught per repo/file so a single failure doesn’t stop the crawl.

* Parallel download repos: `ThreadPoolExecutor` for network/disk I/O.
* Parallel AST parse `ProcessPoolExecutor` for CPU intensive work.

### Dataset size
* Raw data: ~ 20GB with 500 Github repos.
* Processed data: `1,371,223` samples (functions) with the following info: `'repo_name', 'repo_url', 'file_path', 'method_code'`


---

## 2. Pre-training Phase

This stage builds a language model for Python code using the Masked Language Modeling (MLM) objective. 
Including: Tokenizer, MLM dataset and BERTCode training.

### 2.1 Train Tokenizer (BPE)

* Trained a **Byte-Pair Encoding (BPE)** tokenizer from scratch using the collected Python functions.
* Used Hugging Face `tokenizers` library.
* Ensured coverage of special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`, `<mask>`.
* Saved as `python_tokenizer.json` for usage in all later steps.

### 2.2 Prepare Masked Language Modeling (MLM) Dataset

* Each python function is tokenized and splitted into fixed-length sequences (max_length=512).
* Created a dataset MLM:
  * Randomly masked 15% of tokens (BERT-style).
  * Split into:
    * Training: 80%
    * Validation: 10%
    * Test: 10%
* Saved preprocessed splits for later reuse (`train.pt`, `validation.pt`, `test.pt`).

### 2.3 Define and Train Model (BERTCode)

* Use a **BERT-based Transformer** for code pre-training.
* Objective: **Masked Language Modeling (MLM)** — predict masked tokens from surrounding context.
* Model configuration:
  * 6 Transformer layers
  * 8 attention heads
  * Hidden size: 256
  * Intermediate size: 1024
* Trained for several epochs with batch_size = 32, learning_rate `5e-4` using Hugging Face `Trainer`.
* Validation and test accuracy are computed via token-level masked prediction accuracy.

**Output:**

* Final model weights and config saved to `mlm_model_bert/`
* Example performance:

  * Validation loss: *X.XX*
  * Masked-token accuracy: *YY%*

---

## 3. Fine-tuning: 

This stage adapts the pre-trained BERT model to a downstream code understanding task: predicting missing `if` conditions inside Python functions.

### 3.1 Build Fine-tuning Dataset

* From collected functions, identify and masked one `if` condition per function:
  ```python
  if <mask>:
  ```
* Store:
    * Input: function text with `<mask>`
    * Target: the original condition text.
* Saved as JSONL dataset (`train.jsonl`, `validation.jsonl`, `test.jsonl`) with fields: `input` and `target`.

#### 3.2 Fine-tune the Model

* Loaded pre-trained BERT model and the trained tokenizer.
* Replaced each `<mask>` to multiple mask tokens (one per target token).
* Fine-tuned on the “if” dataset using MLM objective.
* Evaluation metrics:
  * Validation loss
  * Masked-token accuracy

**Output:**

* Saved fine-tuned model to: `if_mlm_finetuned/`
* Example metrics:

  * Validation accuracy: *AA%*
  * Test accuracy: *BB%*

---

### 4. Example Inference

**Input:**

```python
def classify(x):
    if <mask>:
        return "positive"
    else:
        return "negative"
```

**Model Prediction:**

```
x > 0
```

---

### 5. Summary

| Stage        | Dataset                | Model               | Goal                        |
| ------------ | ---------------------- | ------------------- | --------------------------- |
| Pre-training | 100k+ Python functions | BERTCode            | Learn general code patterns |
| Fine-tuning  | Masked `if` dataset    | Fine-tuned BERTCode | Predict masked condition    |
