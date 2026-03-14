# Recursive Language Models (RLM) Project Overview

## What is an RLM?
A Recursive Language Model (RLM) is an inference paradigm for expanding the capacity of Large Language Models (LLMs) to handle very long contexts without succumbing to context rot.

Instead of passing the entire document context straight into the model's token window, an RLM places the long prompt in an external environment - specifically as a variable inside a Python REPL (Read-Eval-Print Loop). The root LLM is then allowed to write Python code to iteratively:
- Examine slices of the context
- Decompose the main query into sub-queries
- Recursively call a language model on localized chunks of context

## Core Concepts
- **Root LM (Depth=0):** The primary orchestrator. It receives a query but not the full text. It uses the REPL to write scripts, execute them, and recursively call instances.
- **Recursive sub-calls (Depth>0):** The same base model can be invoked on narrowed chunks of text.
- **Environment (REPL):** Sandbox-like runtime where the full context is kept in RAM.

## Project Idea: Comparing LLM vs RLM
We aim to build a project to explicitly test out the differences and benefits of an RLM architecture versus a traditional vanilla LLM architecture.

**Locked Setup:**
1. **Single base model for everything:** `qwen3.5:2b` via Ollama.
2. **Vanilla path:** same model, direct-context inference.
3. **RLM path:** same model wrapped in REPL-based recursive execution.

Because hardware is constrained (RTX 4050 6GB VRAM), we are intentionally avoiding API providers and keeping execution local and reproducible.

## Expected Features
- Sandbox REPL for Python script execution logic.
- Benchmarking pipeline covering Vanilla Context vs Recursive Context runs.
- Modular code architecture inspired by `alexzhang13/rlm` patterns.
