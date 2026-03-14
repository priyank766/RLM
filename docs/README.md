# RLM Project - Docs Index

> **This `docs/` directory is the project brain.**
> Every decision, research finding, plan, and session log lives here.
> When confused, come back here first.

---

## Files In This Directory

| File | Purpose |
|---|---|
| `README.md` | This index - start here |
| `dev_log.md` | Session-by-session progress log - append each session |
| `tasks.md` | Master task tracker with all phases and todos |
| `project_info.md` | Project objective, goals, and key resources |
| `RLM_Overview.md` | One-page overview: what we're building and why |
| `01_rlm_deep_dive.md` | Full paper breakdown: architecture, algorithm, results, limitations |
| `02_repl_explained.md` | What REPL is, how to build one, REPL loop design |
| `03_hardware_constraints.md` | RTX 4050 6GB reality check - what we can and cannot run |
| `04_implementation_plan.md` | Full project plan: phases, architecture, file structure, design decisions |
| `05_novel_research_ideas.md` | Research extensions and final-phase RL directions |
| `RLM_research_findings.md` | External research audit: what we're doing right, what to fix |
| `implementation_plan.md` | Early lightweight plan (superseded by 04, kept for history) |

---

## Quick Reference: Core Concepts

| Term | Meaning |
|---|---|
| **RLM** | Inference-time wrapper. Context P stored in REPL variable, never in LLM context window |
| **REPL** | Python `exec()`-based env with persistent namespace. LLM writes code cells here |
| **Root LM** | Orchestrator LLM. Sees only metadata about context, writes REPL code |
| **Sub-LM** | Called inside REPL via `llm_query()`. Gets focused sub-chunks. Same model in this project |
| **FINAL()** | LLM terminates the loop by calling this. Returns the answer string |
| **Recursive-first policy** | On long contexts, the runtime forces evidence gathering before finalization |
| **Context rot** | Quality degradation as context length grows - the problem RLM solves |

---

## Quick Reference: Key Paper Results

| Benchmark | Vanilla GPT-5 | RLM (GPT-5) |
|---|---|---|
| OOLONG | baseline | +28.4% |
| OOLONG-Pairs F1 | <0.1% | 58.0% |
| BrowseComp-Plus | baseline | +29% over retrieval |

---

## Quick Reference: Our Stack

```
Language    : Python 3.12
Package mgr : uv
Model       : qwen3.5:2b via Ollama (local only)
Root LM     : qwen3.5:2b
Sub LM      : qwen3.5:2b (same model - fair comparison)
REPL type   : Local exec()-based (LocalREPL)
REPL tools  : chunk_text, keyword_windows, regex_windows, query_chunks, slices
Logging     : JSONL files in experiments/
Hardware    : RTX 4050 Laptop GPU, 6GB VRAM
RL phase    : Final phase via Google Colab + VS Code extension
```

---

## Project Status

| Phase | Status |
|---|---|
| Phase 0 - Research & Planning | **Done** |
| Phase 1 - Core Infrastructure | **Done** |
| Phase 2 - Benchmarks + Runner | **Done** |
| Phase 3 - Run experiments & analysis | In progress |
| Phase 4 - RL-on-RLM | Planned |
| Phase 5 - Publication and public release | Planned |

---

## Current Runtime Notes

- Long contexts now use a recursive-first runtime policy instead of one-shot finalization.
- FINAL() can be rejected on long contexts if the model has not gathered enough evidence.
- The REPL exposes helper functions for chunking, windowing, and focused sub-queries to make real recursive behavior easier.

---

## Public Output Plan

| Output | Goal |
|---|---|
| Medium article | Research-style story of the build, experiments, and lessons |
| GitHub Pages | Public results site with plots, methodology, and benchmark tables |
| Hugging Face release | Public model or adapter so people searching for RLMs can find and inspect the artifact |

---

## Sources

| Resource | Link |
|---|---|
| Paper | https://arxiv.org/abs/2512.24601 |
| Author blog | https://alexzhang13.github.io/blog/2025/rlm/ |
| Official repo | https://github.com/alexzhang13/rlm |
| Minimal impl | https://github.com/alexzhang13/rlm-minimal |
| Community impl | https://github.com/fullstackwebdev/rlm_repl |


