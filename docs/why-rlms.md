# Why Recursive Language Models?

> The problem isn't that LLMs can't read long documents. The problem is that they technically *can* — and the way they "read" fundamentally breaks down at scale.

---

## The Context Window Arms Race (And Why It's a Trap)

If you follow LLM news, you've seen the headlines. Every few months, a new model announces a bigger context window. 128K tokens. 256K. A million. Ten million. It reads like Moore's Law for attention.

But there's a problem nobody puts in the press release: **bigger context windows don't actually solve the underlying issue.**

The issue has a name now — **context rot**. And it's not a bug in any particular model. It's baked into how transformer attention works.

---

## What Is Context Rot?

Context rot is the measurable degradation in an LLM's output quality as the input context grows longer. It happens well before the model hits its maximum context window. It happens on every model that's been tested — all 18 frontier models in one recent survey showed it, from GPT-4.1 to Claude Opus 4 to Gemini 2.5 Pro.

The symptoms are familiar to anyone who's worked with long documents:

- The model misses details buried in the middle of the document
- It gives shallow answers on tasks that require connecting dots across distant sections
- It hallucinates or contradicts itself when the context gets dense
- It "remembers" things at the start and end of the document but forgets everything in between

That last one has a name too: the **lost-in-the-middle effect**. Model attention follows a U-shaped curve across input positions. Information at the beginning (primacy bias) and end (recency bias) gets strong attention. Information in the middle gets substantially less — performance drops by 30% or more for mid-document content.

---

## Why This Happens (The Actual Mechanics)

There are three mechanisms at work, and none of them are going away with a simple scaling fix.

### 1. Attention Dilution

Transformer self-attention distributes a fixed attention budget across all tokens via softmax normalization. As context grows, the denominator in that softmax increases. The per-token attention weight shrinks — roughly proportional to 1/N, where N is the sequence length.

Put simply: the relevant signal doesn't get louder. The noise floor rises.

At 10,000 tokens, a model is tracking approximately 100 million pairwise token relationships. At 100,000 tokens, that's 10 billion. The quadratic scaling ($O(N^2)$) isn't just a computational problem — it's a signal-to-noise problem. The model physically cannot maintain strong focus on target tokens when they're surrounded by exponentially more noise.

### 2. Distractor Interference

Semantically similar but irrelevant tokens actively mislead the model. This gets worse with coherent, logically structured text — because plausible distractors look a lot like the real thing.

Code is especially vulnerable. Consistent naming conventions, repeated patterns, and similar function signatures create high distractor density. If you've ever asked a coding agent to find a specific bug and watched it confidently point at the wrong function, you've seen distractor interference in action.

### 3. Aggregation Failure

Even if a model can retrieve individual facts from a long document, synthesizing those facts into a coherent answer is a separate challenge. The model needs to hold multiple pieces of information in its working context simultaneously while it reasons about their relationships. As context grows, the working context gets crowded. The aggregation step degrades — not because the facts aren't there, but because the model can't juggle them all at once.

---

## Why Standard Fixes Don't Work

### Bigger Context Windows

As we just covered, rot occurs at *every* context length increment, not just at the maximum. A 1M token window still suffers from attention dilution at token 500,000. The problem scales with the window.

### RAG / Retrieval

Retrieval-Augmented Generation tries to fetch only the relevant snippets instead of loading the full document. This works well for single-hop queries — "what year was X founded?" — but falls apart when the answer requires connecting information from multiple distant sections.

The mathematical reason: a single embedding vector has a retrieval ceiling. It can't capture multi-hop relationships that span across different parts of a document. You'd need to retrieve everything and combine it — which puts you back in the aggregation failure trap.

### Summarization / Compaction

Summarizing old context before it fills the window seems sensible until you realize what summarization does: it throws away information. The details that seem unimportant during summarization are often the exact details you need later. This is lossy compression applied to reasoning, and it compounds with every compression cycle.

---

## What an RLM Does Differently

The Recursive Language Model architecture, introduced by Zhang, Kraska, and Khattab in their [2025 paper](https://arxiv.org/abs/2512.24601), takes a step back and asks a different question:

**What if the long prompt was never in the context window at all?**

Instead of treating the context as text the model reads, an RLM treats it as **external state** the model interacts with. Here's how:

### The Context Lives in a REPL

The full document is stored as a Python variable — `context` — in a persistent REPL environment. It's in RAM, not in the model's token window. The model can access any part of it, but only through code it writes itself.

### The Model Sees Only Metadata

When the RLM calls the root LLM, here's what the prompt actually looks like:

```
Context info:
  Total length : 52,719 characters
  Preview      : It is of the first importance to not allow yourself...

Query: What question was Holmes trying to answer?

Write Python code to examine the context and answer the query.
Call FINAL(answer) when you have the answer.
```

That's it. No 50,000-word document. No attention dilution. Just a few lines of metadata and instructions.

### The Model Writes Code to Explore

The LLM then generates Python code. Something like:

```python
# Let's see what we're dealing with
print(f"Context length: {context_len} chars")
print(head(500))
```

The REPL executes this code. The model sees the output — the first 500 characters of the document — and can decide what to do next.

### It Recurses on Focused Sub-Problems

This is where it gets interesting. The REPL has a function called `llm_query()` that the model can call to ask itself questions about specific snippets:

```python
# Search for relevant sections
windows = keyword_windows("question", window=400, limit=5)
for i, w in enumerate(windows):
    print(f"--- Window {i} ---")
    print(w[:200])
    result = llm_query(f"Based on this text: {w}\n\nWhat is the main topic discussed?")
    print(f"Answer: {result}")
```

Each `llm_query()` call is a **fresh invocation of the same model** — but with a tiny, focused prompt. Just the snippet and the question. No dilution. No distractors. The model gets a clean, short context and answers precisely.

### It Terminates When It Has an Answer

When the model is satisfied, it calls `FINAL()`:

```python
FINAL("Holmes was trying to recover a compromising photograph from Irene Adler.")
```

The REPL detects this, captures the answer, and returns it. Done.

---

## Why This Actually Works

### 1. No Passive Context Bloat

The model's working context stays small and focused at all times. There's no attention dilution because there's no massive context to dilute attention across. Every sub-call gets a clean, short prompt.

### 2. Active, Not Passive, Exploration

A vanilla LLM passively receives its entire context and tries to reason over it in one shot. An RLM model actively explores the context through code — searching, filtering, chunking. It's the difference between being handed a 500-page book and told to "find the answer" versus being given a table of contents and the ability to look up specific pages.

### 3. Zero Information Loss

Unlike summarization, nothing is thrown away. The full document sits in memory. The model can access any part of it at any time — it just doesn't have to hold all of it in its working memory simultaneously.

### 4. Deterministic Data Processing

The Python code the model writes uses real string operations — regex, slicing, splitting — that are deterministic and reliable. The model doesn't need to "pay attention" to find a keyword; it uses `re.finditer()`. It doesn't need to estimate distances; it uses `len()` and slicing. The heavy lifting of data access is offloaded to Python, where it belongs.

### 5. Unbounded in Both Directions

The input can be arbitrarily large (it's just a Python string in memory). The output can be arbitrarily large too (the model accumulates results in variables, not in its own token generation). Both constraints that normally bind LLMs — input context window and output token limit — are removed.

---

## What the Paper Found

The results on frontier models were significant:

| Benchmark | Vanilla GPT-5 | RLM (GPT-5) | Gap |
|-----------|--------------|-------------|-----|
| OOLONG | Baseline | +28.4% | Large |
| OOLONG-Pairs F1 | <0.1% | 58.0% | Enormous |
| BrowseComp-Plus | Baseline | +29% over retrieval | Best method |

The OOLONG-Pairs result is the one that really catches your eye. This benchmark requires reasoning over information that's distributed across almost every line of a long document. Vanilla GPT-5 essentially failed completely. The RLM version scored 58%. That's not a marginal improvement — that's a phase change.

The authors also post-trained a model called RLM-Qwen3-8B on RLM trajectories. It outperformed the base Qwen3-8B by 28.3% on average and approached GPT-5 performance on three tasks. This suggests that the RLM approach isn't just a clever prompt trick — it's a genuinely better way to process long contexts, and models can learn to do it better with training.

---

## The Tradeoffs (Because Nothing Is Free)

RLMs aren't strictly better. They're better **for long-context tasks**. For short contexts, the vanilla approach wins:

- **Short contexts (<8K tokens):** The base LLM is faster and slightly more accurate. The RLM's REPL loop adds overhead that isn't justified when the full context fits comfortably in the window.
- **Latency:** RLMs take longer. Multiple REPL iterations, multiple sub-calls — each one is an LLM invocation. On my laptop, a 3-sub-call RLM run takes about a minute. A vanilla pass takes 15 seconds.
- **Code quality dependency:** The RLM's success depends on the model's ability to write working Python code. If the model generates buggy code, it can loop uselessly. This is more likely with smaller models.

The crossover point — where RLM starts winning — is around 16K-32K tokens in my experiments. Below that, vanilla is fine. Above that, RLM pulls ahead because vanilla's context is getting truncated while RLM's isn't.

---

## Where This Is Going

The RLM paper explicitly lists open research directions. The field is young — the paper is from late 2025 and already has multiple community implementations. A few directions I'm exploring in this project:

- **Asynchronous parallel sub-calls**: Current RLMs wait for each `llm_query()` to return before starting the next. Running them in parallel could cut latency dramatically.
- **Training RLMs with Muon optimizer**: The paper used standard AdamW for post-training. Muon, a newer optimizer that applies orthogonalized gradient updates to 2D weight matrices, could converge faster and produce better REPL code generation.
- **Adaptive recursion depth**: Letting the model decide when deeper recursion is worth the computational cost.

But first things first: understanding the basics by running the system on real tasks. Which brings me to the next page — my actual experiment.

---

## Sources

- Zhang, Kraska, Khattab. "Recursive Language Models." arXiv:2512.24601, 2025/2026.
- "Context Rot: Why LLMs Degrade as Context Grows." morphllm.com/context-rot, March 2026.
- "LLM Context Window Management and Long-Context Strategies 2026." zylos.ai, January 2026.
- "Everything You Need to Know About Recursive Language Models." machinelearningmastery.com, March 2026.
- "Recursive Language Models: the paradigm of 2026." primeintellect.ai/blog/rlm, January 2026.
- Bertsch et al. "OOLONG: Long-Context Aggregation Benchmark." arXiv:2511.02817, 2025.
