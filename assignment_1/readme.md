# Project: Pre-training and Fine-tuning Transformer Models for Python Code 

This project automate predict missing `if` condition expressions in Python functions.
Given a code snippet with the condition masked (e.g., `if <mask>:`), the model must generate the correct logical condition.

To achieve this, we leverage the **T5 (Text-to-Text Transfer Transformer)** architecture.
Our approach consists of three major stages:

1. **Data Crawling** – Collect Python code samples from open-source repositories (GitHub).
2. **Pretraining** – Pretraining a T5 model on large Python corpus using a span-corruption (masked language modeling) objective.
3. **Fine-tuning** – Take the pretrained model, further fine-tuning it for the `if`-condition prediction task.
4. **Conclusion** and **Discussion**

Section 4.3 highlights key edge cases - Semantic Mismatch, Context Loss, Syntax Variation, and Loss of Original Purpose - each showing a unique aspect of the model’s code understanding and generation.


**Project Structure:** Below is a list of main components:

| File / Folder                        | Description                                                              |
| ------------------------------------ | ------------------------------------------------------------------------ |
| `cmd_pre_train.sh`                   | Shell script to launch the T5 pre-training process.                      |
| `cmd_fine_tune.sh`                   | Shell script to run model fine-tuning on task-specific datasets.         |
| `craw_python_function.py`            | Script for crawling or collecting Python functions as training data.     |
| `pretrain_t5_span_corruption.py`     | Implements span corruption pre-training strategy for T5.                 |
| `fine_tune_t5_if_condition.py`       | Fine-tuning script specialized for conditional code generation tasks.    |
| `t5_span_collator.py`                | Data collation utilities for masked span prediction during pre-training. |
| `train_tokenizer.ipynb`              | Jupyter notebook for building and training a custom tokenizer.           |
| `processed_data.ipynb`               | Notebook for data preprocessing, cleaning, and formatting.               |
| `evaluation.ipynb`                   | Notebook for evaluating model performance on downstream tasks.           |
| `pretrain_t5.log`, `finetune_t5.log` | Training logs capturing model progress and performance metrics.          |
| `readme.md`                          | Project documentation (this file).                                       |

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

Perfect — here’s a **short, clear description** you can include in your report to explain how the **tokenizer training process** works:

---

## **Tokenizer Training**

Before training the model, we first build a custom tokenizer specialized for Python code.

This is done using SentencePiece, a subword segmentation algorithm that learns how to split code into meaningful tokens (e.g., `if`, `x`, `==`, `None`, `:`).

The overall process:
- Corpus Preparation: all Python functions are combined into a single text file that serves as the training corpus.
- SentencePiece Training: I train a unigram language model with a specified vocabulary size (e.g., 32,000).
   SentencePiece automatically learns subword units without needing tokenized input.
- Integration with Hugging Face: the trained SentencePiece model is wrapped as a T5-compatible tokenizer using the `T5Tokenizer` class Additional, special tokens like `<extra_id_0> ... <extra_id_99>` are added to support span corruption pre-training.
- Finally, the tokenizer is saved in a directory (e.g., `tokenizer_t5_code/`), allowing it to be reused for pre-training and fine-tuning.

This tokenizer allows the model to process raw Python code as a sequence of subword tokens, balancing vocabulary efficiency and code structure understanding.



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


### 3.1 Build Dataset


**Masking if condition**
* For each Python function string, I parse the AST and visit every `ast.If` node (this includes `elif` and nested `if`s).
* For each `if`, I replace the character with the sentinel `<extra_id_0>` (everything else stays intact).
* Result:
  * **Input (source)**:

    ```
    "predict_if_condition: " + def check_function(x):
        if <extra_id_0>>:
            return "positive"
        else:
            return "negative" 
    ```
  * Target (label):
    ```
    x > 0
    ```

**Build dataset**
* The function `build_pairs_from_list` loops all functions, applies the extraction above, and builds pairs (input, output) for each if condition.
* I use `DataCollatorForSeq2Seq` with `label_pad_token_id = -100`, so padded label positions are ignored in the loss.


### 3.2 Fine-tune the Model

**Main Idea**
* Fine-tuning T5 as a seq2seq model, where input = masked function text (with <extra_id_0>) and target = the original condition string.

* Loss is standard cross-entropy between generated tokens and label tokens (padded positions masked as -100).

**How does the model is predicted?**
* The seq2seq model consists: 
    * Encoder input: the entire masked function text.
    * Decoder input: starts with <start_token>, then predicts each next token of the condition until the <end-of-sequence> token.
* For example:
    * If the condition is: ```x > 0 and y < 10```. 
    * The model will predict: ```["x", ">", "0", "and", "y", "<", "10"]```.

**Hyper-parameter**
* Max input sequence lengths: max_src_len=512.
* Max output sequenceL max_tgt_len=128.
* Optimizer "adamw", LR = 5e-5, weight_decay = 0.01. Those are standard T5 fine-tune defaults.
* Train_epochs=5: avoid long training with a large corpus.


**Handling functions that don’t contain `if` statements**

**Training time**: 
* I discard functions that do not contain any `if` statements.
* Rationale: The task is to learn to **reconstruct an `if` condition**. Samples without an `if` cannot produce a (masked_code → condition) pair.
* Data sufficiency: Our corpus yields ~**250k** valid (masked, condition) pairs, so excluding no `if` functions does **not** harm coverage.

**Inference time:**
* Before querying the model, we **check the AST** of the input function.
  * If `if` exists: we mask the `if` and run prediction.
  * If no `if` exists: we return `None` (or raise a clean error), and do not call the model.

**Fine-Tuning Summary**

| Parameter                    | Description                                            | Value                       |
| ---------------------------- | ------------------------------------------------------ | --------------------------- |
| **Training Loss**            | Final training loss after fine-tuning                  | `1.72`                      |
| **Epochs**                   | Number of complete passes through the training dataset | `5`                         |
| **Train time**            | Total time spent on fine-tuning (in seconds)           | ~14.9 hours |


### 3.3. Example Inference

**Input:**

```python
def check_function(x):
    if <mask>:
        return "even"
    else:
        return "odd"
```

**Model Prediction:**

```
x % 2 == 0
```

## 4. Evaluation and discussion

### 4.1. Evaluation

In this section, I will evaluate the fine-tuned model’s performance, we use two main metrics: Exact Match (EM) and BLEU Score.

**Exact Match (EM)**
- This metric measures how the model’s predicted condition exactly matches the ground-truth condition after normalization (removing extra spaces, formatting differences, etc.).
- It provides a strict measurement — a prediction is correct only if it reproduces the exact condition.
- Example:

| True Condition | Predicted Condition | Exact Match |
| -------------- | ------------------- | ----------- |
| `x > 0`        | `x > 0`             | ✅           |
| `x > 0`        | `x >= 0`            | ❌           |

---

**BLEU Score**
- BLEU (Bilingual Evaluation Understudy) measures n-gram overlap between the predicted and true condition strings.
- It captures partial correctness - useful when predictions are semantically close but not identical.
- Example:

| True Condition        | Predicted Condition | BLEU |
| --------------------- | ------------------- | ---- |
| `x > 0 and y < 5`     | `x > 0 and y <= 5`  | 0.85 |
| `isinstance(x, list)` | `type(x) == list`   | 0.62 |


### The table evaluation on test set (10%): 
| Exact Match| 4.4% |
|---|---
| BLEU Score | 28.07 |


### 4.2. Discussion

While both exact match and BLEU Score provide useful quantitative indicators of model performance, both have notable limitations when applied to code prediction tasks such as `<if>` condition generation.

**Exact Match**
- It is overly strict — even a minor formatting difference or logically equivalent variation (e.g., `x > 0` vs. `x>=0`, or extra parentheses) is treated as completely incorrect.
- Thus, Exact Match can underestimate model quality by penalizing predictions that are *semantically correct* but *syntactically different*.

**Example:**

| Gold    | Prediction | EM Result                             |
| ------- | ---------- | ------------------------------------- |
| `x > 0` | `x >= 1`   | ❌ (different literal but same intent) |
| `x > 0` | `(x > 0)`  | ❌ (extra parentheses only)            |

---

**BLEU Score**
- BLEU offer a softer metric than Exact Match. It rewards partial correctness and shared token sequences.
- However, BLEU still fails to capture logical meaning — it only looks at surface-level text similarity.

**Example:**

```python
def check_function(x):
    if <mask>:
        return "positive"
    else:
        return "negative"
```

If the model predicts **`x == 'positive'`** instead of the correct **`x > 0`**,
BLEU may still assign a moderate score because both contain overlapping tokens (`x`, comparison operator, literal).


**Summary:**
* Exact Match: measures syntactic precision but ignores semantic equivalence.
* BLEU: captures partial token overlap but ignores logical correctness.

### 4.3. Edge case 

This section discusses four edge cases that expose how the model interprets and generates code: Semantic Mismatch, Context Loss, Syntax Variation, and Loss of Original Purpose.

**1. Semantic Mismatch Case**

During inference, the model sometimes generates code that differs from the reference label but still preserves the same semantics. 
- Example:

```python
def get_ipython():   
    from IPython.terminal.interactiveshell import TerminalInteractiveShell

    if TerminalInteractiveShell._instance:        
        return TerminalInteractiveShell.instance()
    
    config = tools.default_config()    
    config.TerminalInteractiveShell.simple_prompt = True    
    shell = TerminalInteractiveShell.instance(config=config)
    return shell
```

- True: `TerminalInteractiveShell._instance` 

- Predicted: `tools.default_config() not None`

**Evaluation**

| Metric | Value | Explanation |
|--------|--------|-------------|
| `exact_match` | `False` | Prediction does not exactly match the ground truth string. |
| `bleu_score` | `5.52` | Shows partial token-level similarity. |

---

**Insight:**  
This case highlights the need for **semantic-aware metrics**, such as execution-based testing or Abstract Syntax Tree (AST) similarity, to better capture model understanding.


**2. Contextual Information Loss**

In this case, the model generates syntactically valid Python code but fails to infer the correct conditional expression because it does not provided enough information. 

- Example: 

```python
def unreliable_operation(data: Any) -> Any:
    """Function that might fail and uses the retry decorator."""    import random
    if random.random() < 0.5:
        raise RuntimeError("Random failure")    
    return data
```

- True Output: `random.random() < 0.5`

- Predicted Output: `random.random() < 0`

**Evaluation Summary**

| Metric | Value | Explanation |
|--------|--------|-------------|
| `exact_match` | `False` | The numeric constant `0.5` is replaced by `0`, altering the logic. |
| `bleu_score` | `75.15` | High token similarity despite incorrect condition. |

**Insight:**  
This shows that the model, with limited information input, does not accurately capture the expressed intent in the surrounding context.


**3. Semantic Equivalence with Syntax Variation**

In this case, the model produces an output that is functionally identical to the ground truth but expressed using a different Python syntax pattern.  

For example, the below function `roll_die()`, both implementations achieve the same functional goal — ensuring that the "rolls" entry exists in the state dictionary — but use different logical forms.

- Input:

```python
def roll_die(sides: int, tool_context: ToolContext) -> int:    
    result = random.randint(1, sides)    
    if not "rolls" in tool_context.state:        
        tool_context.state["rolls"] = []    
    tool_context.state["rolls"] = tool_context.state["rolls"] + [result]    
    return result
```                                                                                                                                              
- True Output: `not "rolls" in tool_context.state`

- Predicted: `tool_context.state["rolls"] is None`

**Evaluation**

| Metric                 | Value    | Explanation                                    |
| ---------------------- | -------- | ---------------------------------------------- |
| `exact_match`          | `False`  | The syntax differs from the reference.         |
| `bleu_score`           | `26.27`  | Low token overlap due to syntactic difference. |
| `semantic_equivalence` | ✅ *True* | Functionally identical behavior in Python.     |


**Insight:**
This edge case demonstrates that different valid programming idioms can express the same logic.
Traditional string-based metrics (e.g., Exact Match, BLEU) fail to recognize this, penalizing semantically correct outputs.


**4. Loss original purpose**

In this case, the model correctly predicts a valid and logical condition.
However, the prediction differs from the developer’s original purpose, based on the function’s context.

- Example:
```python
def function(value):
    if x > 0:
        return "positive"
    else:
    return "negative"
```

- True Output: `x > 0`
- Predicted Output: `value == "positive"`

---

**Evaluation Summary**

| Metric | Value | Explanation |
|--------|--------|-------------|
| `exact_match` | `False` | Syntax differs from the ground truth. |
| `bleu_score` | `0.0` | No token overlap, as structure differs completely. |
| `semantic_equivalence` | ✅ *Partially True* | Logically valid, but represents a different interpretation of the function’s goal. |

---

**Insight:**  
This case shows that LLM can produce functionally reasonable output even when diverging from the original task’s purpose.  



---

## 5. Conclusion

This project explored the use of T5 language models for understanding and generating Python code, specifically predicting the masked `if` condition in a function.

The work followed a three-stage pipeline:

1. **Data Collection:**
   Large-scale Python functions were crawled from GitHub and processed into clean training samples.
2. **Pre-training:**
   A base T5 model was further trained on Python code using **span corruption**, helping it learn the syntax and structure of real-world code.
3. **Fine-tuning:**
   The pre-trained model was fine-tuned to predict missing `if` conditions, turning this into a **sequence-to-sequence generation** task.

Model performance was evaluated using **Exact Match** and **BLEU Score** to measure syntactic accuracy and partial similarity.

