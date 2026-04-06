# Code & Setup

> Everything in this project runs locally. No API keys, no cloud dependencies. Just Python, Ollama, and a GPU.

---

## Requirements

- **Python 3.12+**
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **[Ollama](https://ollama.com)** — local LLM runtime
- **GPU**: NVIDIA RTX 4050 (6GB VRAM) recommended, but any GPU that can run Ollama will work

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/priyank766/RLM.git
cd RLM

# 2. Install dependencies
uv sync

# 3. Start Ollama and pull the model
ollama serve
ollama pull qwen3.5:2b

# 4. Run a smoke test (verifies everything works)
uv run python scripts/smoke_test.py
```

The smoke test runs a single 4,000-character task through both the vanilla LLM and the RLM. It should take about 1–2 minutes and produce two correct answers.

---

## Running Benchmarks

### Needle in a Haystack (NIAH)

The classic long-context test — hide a fact in a sea of random words, ask the model to find it.

```bash
# Quick test (4K and 8K contexts)
uv run python scripts/run_niah.py --small

# Full grid (4K, 8K, 16K, 32K)
uv run python scripts/run_niah.py

# Long contexts (32K, 64K, 128K) — where RLM shines
uv run python scripts/run_niah.py --long

# Save full RLM trajectories for training data
uv run python scripts/run_niah.py --save-trajectories

# Resume an interrupted run
uv run python scripts/run_niah.py --run-name niah_full_v2
```

### Full Comparison (NIAH + Long Document QA)

```bash
# Run both benchmark suites
uv run python scripts/run_comparison.py

# Save trajectories
uv run python scripts/run_comparison.py --save-trajectories
```

### Viewing Results

```bash
# Summary table
uv run python scripts/view_results.py experiments/niah_full.jsonl

# Breakdown by context length
uv run python scripts/view_results.py experiments/niah_full.jsonl --by-length

# Show REPL error samples
uv run python scripts/view_results.py experiments/niah_full.jsonl --show-errors
```

### Generating Plots

```bash
# Generate all plot types for a single run
uv run python scripts/plot_results.py experiments/niah_full.jsonl

# Merge multiple runs into one comparison
uv run python scripts/plot_results.py experiments/niah_v1.jsonl experiments/niah_v2.jsonl --merge
```

Plot types generated:
- `_accuracy_by_context.png` — accuracy vs context length
- `_latency_by_context.png` — latency scaling
- `_advantage_by_context.png` — RLM accuracy minus vanilla accuracy
- `_outcome_decomposition.png` — stacked bar: both correct, RLM-only, vanilla-only, both wrong
- `_niah_accuracy_pos_10.png` — accuracy at 10% needle position
- `_niah_accuracy_pos_50.png` — accuracy at 50% needle position
- `_niah_accuracy_pos_90.png` — accuracy at 90% needle position
- `_rlm_iterations.png` — average iterations by context length
- `_rlm_subcalls.png` — average sub-calls by context length
- `_rlm_failure_rates.png` — timeout and error rates

---

## Project Structure

```
RLM/
├── src/
│   ├── rlm/                        # Core RLM engine
│   │   ├── clients/
│   │   │   ├── base.py             # Abstract LLM client interface
│   │   │   └── ollama.py           # Ollama HTTP client
│   │   ├── repl.py                 # LocalREPL — exec-based REPL
│   │   └── rlm_repl.py             # RLM_REPL — full recursive loop
│   ├── baseline/
│   │   └── vanilla_llm.py          # VanillaLLM — direct context inference
│   ├── benchmarks/
│   │   ├── niah.py                 # Needle in a Haystack generator
│   │   └── long_doc_qa.py          # Long Document QA generator
│   ├── comparison/
│   │   ├── runner.py               # Runs both methods, saves JSONL
│   │   └── eval.py                 # Scoring utilities
│   └── training/
│       └── muon.py                 # Muon optimizer (Newton-Schulz)
│
├── scripts/
│   ├── smoke_test.py               # Quick single-task verification
│   ├── run_niah.py                 # NIAH benchmark suite
│   ├── run_comparison.py           # Full comparison (NIAH + LongDocQA)
│   ├── plot_results.py             # Research-style plots
│   ├── view_results.py             # Results analysis
│   ├── build_sft_dataset.py        # Trajectory → SFT training data
│   ├── train_sft.py                # QLoRA SFT with AdamW or Muon
│   └── eval_sft.py                 # Evaluate SFT model
│
├── prompts/
│   └── rlm_system.txt              # System prompt for root LLM
│
├── experiments/                    # Results, checkpoints, plots
│   ├── plots/                      # Generated PNG charts
│   ├── checkpoints/                # SFT model checkpoints
│   └── *.jsonl                     # Benchmark results
│
├── docs/                           # All project documentation
└── mkdocs.yml                      # This site's config
```

---

## Key Files

### `src/rlm/repl.py` — The REPL

The exec-based Python REPL with persistent namespace. Stores the full context as a variable, injects helper functions, protects built-ins from being overwritten.

### `src/rlm/rlm_repl.py` — The RLM Loop

The main recursive loop. Root LLM generates code, REPL executes it, metadata feedback goes back to root LLM, repeat until FINAL() or timeout. Includes the recursive-first policy for long contexts.

### `src/rlm/clients/ollama.py` — The Ollama Client

HTTP client for the local Ollama server. Thinking mode disabled for speed. 600-second timeout to handle slow multi-turn RLM calls.

### `src/baseline/vanilla_llm.py` — The Vanilla Baseline

Direct context inference. Full document passed into the model's context window (truncated if it exceeds the limit). No REPL, no recursion.

### `src/benchmarks/niah.py` — NIAH Generator

Generates Needle in a Haystack tasks with configurable context lengths, needle positions, and random seeds. Uses common English words for the haystack.

### `src/benchmarks/long_doc_qa.py` — LongDocQA Generator

Generates coherent filler text with embedded facts. More realistic than NIAH — tests reasoning over coherent text, not random words.

### `src/comparison/runner.py` — The Runner

Runs both vanilla and RLM on the same tasks. Resumable (fingerprint-based skip). Saves JSONL results. Optional trajectory saving for training data.

### `src/comparison/eval.py` — Scoring

String-based evaluation: exact match, contains-answer, numeric match. No neural judges needed.

---

## Training Pipeline (Phase 4)

```bash
# Step 1: Build SFT dataset from collected trajectories
uv run python scripts/build_sft_dataset.py

# Step 2: Stop Ollama (frees VRAM for training)
ollama stop qwen3.5:2b

# Step 3a: Train with AdamW (baseline)
uv run python scripts/train_sft.py --optimizer adamw --epochs 3

# Step 3b: Train with Muon (ablation)
uv run python scripts/train_sft.py --optimizer muon --epochs 3

# Step 4: Evaluate SFT model
uv run python scripts/eval_sft.py experiments/checkpoints/sft_adamw/final
```

---

## Configuration

All configuration lives in code. Key parameters you might want to change:

| Parameter | Default | Where |
|-----------|---------|-------|
| Model | `qwen3.5:2b` | `OllamaClient.__init__()` |
| Ollama URL | `http://localhost:11434` | `OllamaClient.__init__()` |
| Temperature | `0.0` | `OllamaClient.__init__()` |
| Max REPL iterations | `20` | `RLM_REPL.__init__()` |
| Long context threshold | `16,000` chars | `RLM_REPL.__init__()` |
| Thinking mode | `False` | `OllamaClient.__init__()` |
| Timeout | `600` seconds | `OllamaClient.__init__()` |

---

## Troubleshooting

### Ollama not available
```
ERROR: Ollama not available at http://localhost:11434
```
Start Ollama: `ollama serve`  
Pull the model: `ollama pull qwen3.5:2b`

### RLM timeouts on first call
Qwen models with thinking mode enabled can take 30–120 seconds on the first call. Make sure `think=False` is set in the OllamaClient.

### Unicode errors on Windows
The plotting scripts use UTF-8 encoding. If you get `UnicodeEncodeError` on Windows, set your terminal to UTF-8: `chcp 65001`

### GPU memory errors
If Ollama can't load the model, close other GPU applications. The 2B model needs about 3–4GB of VRAM.

---

*All results are reproducible. Same seeds, same model, same hardware = same outputs.*
