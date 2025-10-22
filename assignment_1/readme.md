# Project: Pre-training and Fine-tuning Transformer Models for Python Code 

This project automate predict missing `if` condition expressions in Python functions.
Given a code snippet with the condition masked (e.g., `if <mask>:`), the model must generate the correct logical condition.

To achieve this, we leverage the **T5 (Text-to-Text Transfer Transformer)** architecture.
Our approach consists of three major stages:

1. **Data Crawling** – Collect Python code samples from open-source repositories (GitHub).
2. **Pretraining** – Pretraining a T5 model on large Python corpus using a span-corruption (masked language modeling) objective.
3. **Fine-tuning** – Take the pretrained model, further fine-tuning it for the `if`-condition prediction task.

## 1. Data Crawling and Tokenization


### 1.1. Data Crawling
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


## 1.2. Tokenization

# TODO


---

## 2. Pre-training Phase

This step pretrain a T5 model (t5-small) on Python functions using the T5 span corruption objective.
It adapts T5 to code syntax before downstream fine-tuning (e.g., predicting <if> conditions).

### 2.1. Prepare  Dataset

I start with the large collection of Python functions, crawled from GitHub and stored in `dataset/processed/processed_data.csv`. 
Each entry is a valid function (e.g., 10–50 lines) extracted using the Python `ast` parser.

Each function is treated as one training example.
During pretraining, we will corrupt each function by masking random spans of text and asking the model to reconstruct them.

#### 2.1.1 Span Corruption
- T5 does not use a traditional token-level "mask 15%" like BERT. Instead, it uses span corruption, where contiguous spans of tokens are replaced by sentinel tokens like <extra_id_0>, <extra_id_1>, etc.

Suppose we have this Python function:
```python
def add(x, y):
    return x + y
```

During pretraining, a random span (e.g., `x + y`) is replaced with a sentinel:
**Input (corrupted):**
```
def add(x, y):
    return <extra_id_0>
```

**Target (reconstruction):**
```
<extra_id_0> x + y
```
The model learns to fill in the masked code.


#### 2.1.2. On-The_Fly Masking
The script performs **dynamic masking** using a *collator* for each training step:
* Randomly selects ~15% of tokens (`noise_density = 0.15`).
* Groups them into spans of average length ~3 tokens (`mean_span_length = 3.0`).
* Replaces each span with a sentinel token.
* Builds the corresponding target sequence containing the removed spans.

This ensures the model sees new corruptions each epoch, improving generalization.



### 2.2. Define and Train Model (T5)
Tokenizer and Model Initialization

1. Tokenizer: I use a custom SentencePiece tokenizer trained on Python code. It supports the `<extra_id_*>` tokens required for span corruption. (section 1.2)
2. Base Architecture: Start from `t5-small` — a pre-trained language model.


Model Input/Output
* Input: Corrupted function text with sentinels (<mask>).
* Target: The concatenated masked spans, each prefixed by its sentinel.
* Loss: Standard sequence-to-sequence cross-entropy between the predicted and target tokens.


Hyper-parameter:
* **Optimizer:** Adafactor (memory-efficient, works well with T5)
* **Learning Rate:** 1e-3
* **Epochs:** 3–5
* **Objective:** Minimize reconstruction loss — make the model better at filling in missing spans of code.


After training, I get a folder output trained model:

```
t5_code_pretrained/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── spiece.model
└── special_tokens_map.json
```

This folder is now a completed T5 model, ready to be fine-tuned on downstream tasks.


### 2.3. Evaluation

The table below is the summarize of T5 pre-train process.

**Pretraining Evaluation Summary:**

| Metric                              | Description                                                         |          Value          |
| :---------------------------------- | :------------------------------------------------------------------ | :---------------------: |
| **Average Training Loss (overall)** | Mean loss across all epochs                                         |        **2.0674**       |
| **Final Loss**             | Cross-entropy loss on evaluation set |        **1.7522**       |
| **Epochs Completed**                | Total number of training epochs                                     |          **5**          |
| **Total Runtime**                   | Training time (seconds)                                  | **~5.2 hours** |
| **Output Checkpoint**               | Directory of the continued-pretrained model                         |  `t5_code_pretrained/`  |


---

## 3. Fine-tuning: 


### 3.1 Build Fine-tuning Dataset



#### 3.2 Fine-tune the Model


**Output:**



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