# RLM — Recursive Language Models

> What happens when you stop stuffing entire documents into an LLM's context window — and instead give it a programming environment to work with?

---

## The Short Version

Standard LLMs get worse the more context you give them. This isn't a bug in any particular model — it's baked into how transformer attention works. The phenomenon has a name now: **context rot**.

Recursive Language Models (RLMs) take a different approach. Instead of passing a 50,000-word document straight into the model and hoping it finds the needle, an RLM stores that document as a variable in a Python REPL. The model then writes code to search, chunk, and query slices of it — recursively calling itself on focused sub-problems. The full context never enters the model's token window. Only metadata does.

Same model. Different architecture. Measurably different results on long contexts.

This project is my attempt to build that system from scratch, run it on my laptop GPU (an RTX 4050 with 6GB of VRAM — not exactly a research cluster), and produce clean, reproducible comparisons between vanilla inference and recursive inference.

---

## What Prompted This

The paper [Recursive Language Models](https://arxiv.org/abs/2512.24601) by Zhang, Kraska, and Khattab came out in late 2025. The results were striking — on benchmarks like OOLONG, an RLM-wrapped GPT-5 outperformed vanilla GPT-5 by 28.4%. On OOLONG-Pairs, which requires reasoning across almost every line of a long document, vanilla GPT-5 scored below 0.1% F1. The RLM version scored 58.0%.

That gap is hard to ignore.

What interested me most was the elegance of the idea. There's no new model architecture being trained. No fancy new attention mechanism. No larger context window. The trick is architectural: treat the long prompt as **external state** that the model interacts with through code, rather than text it has to read in one pass.

I wanted to understand this properly. And the best way to understand something is to build it yourself.

---

## What This Project Does

```
Vanilla LLM:
  [system prompt + FULL 15,000-word document + query] → LLM → answer
  (attention dilution kicks in — quality degrades)

RLM:
  [system prompt + metadata: "document is 52,719 chars"] → root LLM → writes Python code
                                                                               |
  REPL executes code. The full document lives in a `context` variable in RAM.
  Model writes: windows = keyword_windows("festival"); results = [llm_query(w) for w in windows]
                                                                               |
  Sub-LM answers focused questions on small, clean text snippets (~800 chars each)
                                                                               |
  Root LLM consolidates the evidence → FINAL(answer)
  Total sub-calls: 3. Total time: under a minute. Full document searched.
```

The comparison is deliberately fair: **the same model** runs both paths. Same weights, same tokenizer, same temperature. The only variable is the inference architecture.

---

## Key Results So Far

On a 52,719-character document (roughly 9,200 words from "A Scandal in Bohemia") with my local Qwen 2B model capped at a 4,096-token context window:

| Metric | Vanilla LLM | RLM |
|--------|------------|-----|
| Context seen | Truncated to ~4,096 tokens | Full 52,719 chars |
| Sub-calls needed | N/A (single pass) | **3** |
| Answer quality | Depends on where the answer sits in the truncation window | Consistent — searches the full document |

The real story isn't just accuracy — it's **how** the RLM gets there. The model doesn't read the whole document. It reads metadata about the document, writes code to find interesting sections, asks itself focused questions about those sections, and stitches together an answer. Three sub-calls. That's it.

---

## How to Navigate This Site

| Page | What you'll find |
|------|-----------------|
| **[Why RLMs →](why-rlms.md)** | The problem of context rot, why bigger context windows don't fix it, and why RLMs are a genuinely different approach |
| **[How It Works →](how-it-works.md)** | Step-by-step walkthrough of the RLM architecture, the REPL loop, recursive sub-calls, and how the model actually thinks through code |
| **[My Experiment →](experiment.md)** | The actual run I did on my hardware — the document, the question, the trajectory, and what happened at each REPL iteration |
| **[Code & Setup →](code-setup.md)** | How to clone, install, and run benchmarks yourself. Full project structure and CLI commands |
| **Research Notes** | Deeper dives into the paper, external research findings, novel extensions (Muon optimizer, async sub-calls), and the SFT training pipeline audit |

---

## The Stack

| Component | What I'm using |
|---|---|
| Language | Python 3.12 |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Model | `qwen3.5:2b` via [Ollama](https://ollama.com) |
| REPL | Local `exec()`-based Python |
| GPU | RTX 4050 Laptop (6GB VRAM) |
| Training | QLoRA (4-bit NF4) + LoRA rank 16 |

Everything runs locally. No API keys, no cloud dependencies, no hidden costs.

---

## Background

This work is based on the paper [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) by Zhang, Kraska & Khattab. The official implementation lives at [alexzhang13/rlm](https://github.com/alexzhang13/rlm).

What you'll find here is my own from-scratch build, my own experiments, and my own analysis. The goal isn't to reproduce their results — it's to understand the idea deeply enough to contribute something of my own.

---

*Built on a laptop. All results reproducible with the same setup.*
