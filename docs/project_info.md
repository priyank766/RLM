# Project Output and Goals
Project exploring Recursive Language Models (RLM), specifically evaluating the concepts presented in the paper [Recursive Language Models (arXiv 2512.24601)](https://arxiv.org/abs/2512.24601) by Alex L. Zhang, Tim Kraska, and Omar Khattab.

## Objective
To develop a small-scale framework to directly compare the capabilities of standard LLMs with RLMs on long-context processing. Instead of training from scratch (which is compute expensive and requires larger GPUs), we will use one local open model only: `qwen3.5:2b` via Ollama, and compare vanilla inference vs recursive inference on the same model.
The goal is NOT to build a fancy UI, but to obtain verifiable comparison metrics between the two inference paradigms (Vanilla vs Recursive).
Goal is to showcase our ability to read and implement research papers, understand how RLM works, and share practical results with the community.

## What are RLMs?
Standard LLMs suffer from "context rot" - they struggle to accurately reason over extremely long contexts despite technically large token windows.
RLMs shift the paradigm. Rather than forcing the LLM to process thousands of tokens in full context simultaneously, RLMs provide the LLM with a programmable interface (a Read-Eval-Print Loop, commonly a Python REPL). The model searches, extracts, summarizes, and reduces snippets recursively within this environment until it arrives at an answer.

## Key Resources
- Arxiv Paper: https://arxiv.org/abs/2512.24601
- Blog Post: https://alexzhang13.github.io/blog/2025/rlm/
- Codebases to analyze:
  - https://github.com/fullstackwebdev/rlm_repl
  - https://github.com/alexzhang13/rlm
