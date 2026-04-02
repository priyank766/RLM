# RLM — Recursive Language Model vs Vanilla LLM

A local, reproducible comparison framework that puts **Recursive Language Models (RLMs)** head-to-head against standard LLM inference on long-context tasks — using the exact same model for both. Includes **SFT training with QLoRA** and **Muon optimizer ablation**.

> Based on the research paper: [Recursive Language Models (arXiv:2512.24601)](https://arxiv.org/abs/2512.24601) by Zhang, Kraska & Khattab (2025/2026)

---

## What is an RLM?

Standard LLMs suffer from **context rot** — the longer the document you feed them, the worse their reasoning gets. Even models with 128K token windows degrade badly on information-dense tasks.

An RLM fixes this by **never putting the full document in the LLM's context window**. Instead:

1. The document is stored as a Python variable (`context`) in an external REPL
2. The LLM writes Python code to examine slices of it
3. The LLM calls itself recursively via `llm_query()` on focused sub-chunks
4. When it has the answer, it calls `FINAL(answer)` to terminate

```
Vanilla LLM:
  [system prompt + FULL 32K document + query] -> LLM -> answer
  (context rot degrades quality above ~8K tokens)

RLM:
  [system prompt + metadata about document + query] -> root LLM -> Python code
                                                                        |
  REPL executes code, `context` variable holds full document in RAM
  LLM writes code like: chunks = [...]; results = [llm_query(chunk) for chunk in chunks]
                                                                        |
  Sub-LM answers focused questions on small, clean chunks
                                                                        |
  Root LLM consolidates -> FINAL(answer)
```

**Same model. Completely different architecture. Measurably different results.**

---

## Key Results

| Context | Vanilla | RLM | Winner |
|---------|---------|-----|--------|
| 4K-8K   | 100%    | 100%| Tie (both fit in KV cache) |
| 16K     | 100%    | 67% | Vanilla (RLM code quality issue) |
| 32K     | 33%     | 33% | Tie (KV cache truncation begins) |
| 64K     | 33%     | 67% | **RLM** (vanilla is blind, RLM finds it) |
| 128K    | 0%      | --  | **RLM** (vanilla completely fails) |

> RLM advantage emerges at 32K+ where Ollama's KV cache truncates vanilla's context. RLM is unaffected because it searches the full document via REPL code.

---

## Project Structure

```
RLM/
|-- src/
|   |-- rlm/                        # Core RLM engine
|   |   |-- clients/
|   |   |   |-- base.py             # BaseLLMClient (abstract)
|   |   |   |-- ollama.py           # OllamaClient -- HTTP to Ollama
|   |   |-- repl.py                 # LocalREPL -- exec-based REPL
|   |   |-- rlm_repl.py             # RLM_REPL -- full recursive loop
|   |-- baseline/
|   |   |-- vanilla_llm.py          # VanillaLLM -- direct context inference
|   |-- benchmarks/
|   |   |-- niah.py                 # Needle in a Haystack generator
|   |   |-- long_doc_qa.py          # Long Document QA generator
|   |-- comparison/
|   |   |-- runner.py               # Run vanilla + RLM, save JSONL
|   |   |-- eval.py                 # Score results
|   |-- training/                   # Phase 4: SFT + RL training
|       |-- muon.py                 # Muon optimizer (Newton-Schulz)
|
|-- scripts/
|   |-- smoke_test.py               # Quick single-task test
|   |-- run_niah.py                 # NIAH benchmark suite
|   |-- run_comparison.py           # Full suite (NIAH + LongDocQA)
|   |-- plot_results.py             # DeepSeek-R1-Zero style plots
|   |-- view_results.py             # Analyse saved JSONL results
|   |-- build_sft_dataset.py        # Trajectory -> SFT training data
|   |-- train_sft.py                # QLoRA SFT with AdamW or Muon
|   |-- eval_sft.py                 # Evaluate SFT model vs base
|
|-- prompts/
|   |-- rlm_system.txt              # System prompt for root LM
|
|-- experiments/                    # Results, checkpoints, plots
|   |-- plots/                      # Generated PNG charts
|   |-- checkpoints/                # SFT model checkpoints
|   |-- *.jsonl                     # Benchmark results
|   |-- *_trajectories.jsonl        # RLM code traces for training
|   |-- sft_dataset.jsonl           # Filtered SFT training data
|
|-- docs/                           # Research, decisions, logs
|   |-- dev_log.md                  # Session-by-session progress
|   |-- tasks.md                    # Master task tracker
|   |-- 01_rlm_deep_dive.md        # Paper breakdown
|   |-- 02_repl_explained.md       # REPL concept + build guide
|   |-- 03_hardware_constraints.md # RTX 4050 reality check
|   |-- 04_implementation_plan.md  # Architecture + phase plan
|   |-- 05_novel_research_ideas.md # Future directions
|
|-- pyproject.toml
|-- .python-version                 # Python 3.12
```

---

## Hardware & Model

| | |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop GPU (6GB VRAM) |
| Model | `qwen3.5:2b` via Ollama (inference) / `Qwen2.5-1.5B` via HF (training) |
| Root LM | `qwen3.5:2b` |
| Sub LM | `qwen3.5:2b` (same model) |
| REPL | Local `exec()`-based Python |
| Training | QLoRA (4-bit NF4) + LoRA rank 8 |

---

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com)

```bash
# 1. Clone / enter the project
cd RLM

# 2. Install base dependencies
uv sync

# 3. Start Ollama and pull model
ollama serve
ollama pull qwen3.5:2b

# 4. (Optional) Install training dependencies for Phase 4
uv pip install -e ".[train]"
```

---

## Usage

### Benchmarking (Phase 2-3)

```bash
# Smoke test — verify everything works (~1-2 min)
uv run python scripts/smoke_test.py

# NIAH benchmark — short contexts
uv run python scripts/run_niah.py --small

# NIAH benchmark — full grid (4K/8K/16K/32K)
uv run python scripts/run_niah.py --save-trajectories

# Long-context NIAH (32K/64K/128K) — where RLM shines
uv run python scripts/run_niah.py --long --save-trajectories

# Full comparison (NIAH + LongDocQA)
uv run python scripts/run_comparison.py --save-trajectories

# Resume an interrupted run (fingerprint-based skip)
uv run python scripts/run_niah.py --run-name niah_full_v2

# Generate plots (DeepSeek-R1-Zero style)
uv run python scripts/plot_results.py experiments/niah_full_v2.jsonl

# Merge multiple runs into one plot set
uv run python scripts/plot_results.py experiments/niah_full_v2.jsonl experiments/niah_long_v1.jsonl --merge
```

### Training (Phase 4)

```bash
# Step 1: Build SFT dataset from collected trajectories
uv run python scripts/build_sft_dataset.py

# Step 2: STOP Ollama (frees VRAM for training)
ollama stop qwen3.5:2b

# Step 3a: Train with AdamW (baseline)
uv run python scripts/train_sft.py --optimizer adamw --epochs 3

# Step 3b: Train with Muon (ablation)
uv run python scripts/train_sft.py --optimizer muon --epochs 3

# Step 4: Evaluate SFT model
uv run python scripts/eval_sft.py experiments/checkpoints/sft_adamw/final
uv run python scripts/eval_sft.py experiments/checkpoints/sft_muon/final

# Step 5: Compare all results
uv run python scripts/plot_results.py experiments/eval_sft_adamw.jsonl experiments/eval_sft_muon.jsonl --merge

(optional)RLM Visualization Server — run RLM and watch every step in your browser.
uv run python scripts/run_viz.py
uv run python scripts/run_viz.py --port 8765
```

### Training Parameters (6GB VRAM)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Quantization | 4-bit NF4 (QLoRA) | Fits 2B model in ~3.5GB |
| LoRA rank | 8 | Low memory, sufficient for SFT |
| LoRA alpha | 16 | Standard 2x rank |
| Batch size | 1 | Minimum for 6GB |
| Grad accumulation | 4 | Effective batch = 4 |
| Max seq length | 2048 | Caps memory usage |
| Optimizer | AdamW or Muon | Muon for 2D weights only |

---

## Muon Optimizer

[Muon](https://github.com/KellerJordan/Muon) is an optimizer designed for weight matrices in transformers. Instead of standard gradient descent, it applies **Newton-Schulz orthogonalization** to compute a spectrally-normalized update direction.

**How it works:**
1. Accumulate gradient with Nesterov momentum
2. Apply Newton-Schulz iterations to approximate the polar decomposition: `G -> U @ V^T`
3. This gives a "direction-only" update (strips out magnitude)
4. Update: `W -= lr * orthogonal_direction`

**Split strategy:**
- **Muon**: all 2D weight matrices (attention Q/K/V/O projections, MLP layers)
- **AdamW**: everything else (embeddings, layernorm, biases)

**Ablation question:** Does Muon converge faster and produce better RLM code trajectories compared to standard AdamW?

---

## How RLM Works (in code)

```python
from rlm.clients.ollama import OllamaClient
from rlm.rlm_repl import RLM_REPL

client = OllamaClient(model="qwen3.5:2b")
rlm = RLM_REPL(root_client=client, max_iterations=20)

result = rlm.completion(
    context="...your very long document...",
    query="What is the activation code mentioned in section 3?"
)

print(result["answer"])       # The model's answer
print(result["iterations"])   # How many REPL turns it took
print(result["sub_calls"])    # How many llm_query() calls were made
print(result["trajectory"])   # Full step-by-step code + output log
```

---

## Benchmarks

### Needle in a Haystack (NIAH)
- **Context**: random English words (haystack)
- **Needle**: one specific fact inserted at a controlled position (10%, 50%, 90%)
- **Query**: ask for that exact fact
- **Why**: tests pure retrieval across increasing context lengths

### Long Document QA
- **Context**: filler sentences with 5 facts embedded throughout
- **Query**: ask about one specific fact buried in filler
- **Why**: more realistic than NIAH — coherent text, not random words

---

## SFT Training Pipeline

```
Benchmark runs (--save-trajectories)
        |
        v
*_trajectories.jsonl  (raw RLM code traces)
        |
        v
build_sft_dataset.py  (filter: correct + FINAL called + no errors)
        |
        v
sft_dataset.jsonl     (chat-format messages for SFT)
        |
        v
train_sft.py          (QLoRA + AdamW or Muon)
        |
        v
checkpoints/sft_*/    (LoRA adapters)
        |
        v
eval_sft.py           (run SFT model through NIAH, compare to base)
```

---

## Roadmap

- [x] Phase 0 -- Research & planning
- [x] Phase 1 -- Core infrastructure (REPL, RLM loop, Ollama client, vanilla baseline)
- [x] Phase 2 -- Benchmarks (NIAH, Long-Doc QA, comparison runner, eval, plots)
- [x] Phase 2.5 -- Resumable runs, trajectory saving, DeepSeek-style plots
- [/] Phase 3 -- Analysis (NIAH complete, LongDocQA in progress)
- [/] Phase 4a -- SFT Training with QLoRA (pipeline built, training pending)
- [ ] Phase 4b -- GRPO Reinforcement Learning
- [ ] Phase 4c -- Muon Optimizer Ablation (AdamW vs Muon)
- [ ] Phase 5 -- Publication (Medium article, GitHub Pages)

See `docs/tasks.md` for the full checklist and `docs/dev_log.md` for session history.

---

## Research Background

| Resource | Link |
|---|---|
| RLM Paper (arXiv) | https://arxiv.org/abs/2512.24601 |
| Author blog post | https://alexzhang13.github.io/blog/2025/rlm/ |
| Official implementation | https://github.com/alexzhang13/rlm |
| Muon optimizer | https://github.com/KellerJordan/Muon |
| Unsloth (QLoRA training) | https://github.com/unslothai/unsloth |
