

# Evaluate Code Generation of LLM using Vector Representation

Predicting whether a piece of code will pass a given test case without executing it, by working in embedding space.

## LLM generate code

**Problem definition**
We have: 
- A natural language describe the task.
- Function signature.
- An LLM.

The LLM takes a text prompt and outputs a code snippet



Using LLM to generate code. 
List of LLM:
- GPT4o
- Llama 3.1 (8B)
- Qwen 2.5 (7B)
- Deepseek Coder (7B)


## Code Evaluation using Vector Representation

Given:
- Code snippet $c$
- Test case $t$

We compute the embedding: 
- $E_c = f(c)$
- $E_t = f(t)$ 

The goal is to train the model $P(\text{pass}|E_c, E_t)$, that predict the probability that code $c$ will pass the test case $t$.

### Feature Engineering (Embedding Fusion)
Given that we have 2 vector $E_c$ and $E_t$, we must combine code + test embeddings into a joint feature vector.


## Dataset
Super simple + clean + small → MBPP

Standard + widely used → HumanEval

Medium difficulty + thousands of samples → APPS or MultiPL-E

Data-science tasks → DS-1000
