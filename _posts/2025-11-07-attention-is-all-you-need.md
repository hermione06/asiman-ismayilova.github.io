---
layout: post
title: "Understanding 'Attention Is All You Need'"
date: 2025-11-07
categories: [NLP, Transformers]
---

**Paper:** [Vaswani et al., 2017 — Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

### 🧩 Why I Chose This Paper
This paper introduced the **Transformer architecture**, which became the foundation for almost every modern language model (like BERT, GPT, and T5).  
I wanted to understand how it replaced RNNs and LSTMs with a more efficient mechanism.

---

### 💡 Problem It Solves
Before Transformers, sequence models like RNNs and LSTMs struggled with **long-term dependencies** and **slow training**.  
This paper proposed using **self-attention** to capture relationships between all words in a sequence simultaneously — no recurrence needed.

---

### 🔍 Core Idea
The key concept is **self-attention**, where each word looks at every other word in a sentence to decide what’s important.  

For example:
> In “The cat sat on the mat,”  
> the word *“cat”* attends to *“sat”* and *“mat”* to understand the sentence meaning.

The model uses:
- **Multi-head attention** → multiple perspectives at once  
- **Positional encoding** → keeps word order info  
- **Encoder-decoder structure** → processes and generates sequences

---

### 📊 Results
Transformers achieved **state-of-the-art performance** on translation tasks and trained **much faster** than RNNs.

---

### ✏️ My Takeaways
- Self-attention is both simple and powerful.  
- Removing recurrence made it easier to scale models massively.  
- Reading this paper helped me see why Transformers dominate NLP today.

---

### 🔗 Further Reading
- *“The Illustrated Transformer”* by Jay Alammar  
- *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)*  
