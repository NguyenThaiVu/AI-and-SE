# Lab-01: Recommending Code Tokens via N-gram Models

**Course:** AI4SE 2025 – Prof. Antonio Mastropaolo
**Academic Year:** 2025/2026

---

## 1. Overview

This lab implements a **probabilistic N-gram language model** for code completion.
The model learns token sequence probabilities from a training corpus of Java methods, then predicts the most likely next tokens.

Training process:
* Training with multiple **N values** (3, 5, 7).
* Evaluation using **perplexity** as the intrinsic metric.
* Sampling 1000 contexts from the test set and store in `lab-01/output_n_gram/predictions.jsonl`

---

## 2. Dataset

We use the pre-crawled dataset provided in Lab-00:

* **Train set:** `java_method_train.csv`
* **Test set:** `java_method_test.csv`
* Each row contains a `code_tokens` column (tokenized Java methods).

---

## 3. Implementation

* Key components:
  * **N-gram:** Supports configurable N and add-k smoothing.
  * **Evaluation:** Perplexity on validation and test sets.
  * **Sampling:** Generates completions until `<EOS>`, `}` or length 50.

* Output files:
  * `metrics.csv` — validation & test perplexities.
  * `best_model.json` — selected N and smoothing config.
  * `predictions.jsonl` — ≥1000 test contexts with top-k predictions + sampled completion.

---

## 🚀 How to Run

```bash
python n_gram.py \
  --train_csv java_method_train.csv \
  --test_csv java_method_test.csv \
  --n_list 3 5 7 \
  --k 0.1 \
  --topk 10 \
  --num_samples 1000 \
  --out_dir output_n_gram \
```

Check progress with:

---

## 4. Results

### Validation Perplexities

| N | val\_perplexity |
| - | --------------- |
| 3 | **385.63**      |
| 5 | 2100.27         |
| 7 | 4917.68         |

➡️ **Best validation perplexity** is achieved at **N=3**.

### Test Perplexity

| Metric           | Value  |
| ---------------- | ------ |
| TEST\_PERPLEXITY | 393.92 |

---

## 5. Conclusion

* The **trigram model (N=3)** performs best, giving the lowest perplexity on validation and test data.
* This confirms that smaller N is more reliable for limited training data.
* The generated `predictions.jsonl` demonstrates the model’s ability to produce realistic next-token suggestions and full sampled completions.

---

## 6. Deliverables

* Code: `n_gram.py`
* Outputs: `metrics.csv`, `best_model.json`, `predictions.jsonl` (≥1000 samples)
* Report: `README.md` (this file)
