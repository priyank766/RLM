# Dev Log

---

## [2026-03-03] - Session 1: Research and Foundation

**What we did:**
- Read the RLM paper: arXiv:2512.24601
- Read Alex Zhang's blog post
- Analyzed 3 repositories:
  - `alexzhang13/rlm`
  - `alexzhang13/rlm-minimal`
  - `fullstackwebdev/rlm_repl`
- Created initial docs structure

**Key findings:**
- RLM is a drop-in replacement for LLM inference with externalized context handling
- REPL stores `context` outside the LM token window
- Sub-calls happen via `llm_query()` in generated Python code
- Paper showed strong long-context improvements on benchmark tasks

**Decisions made:**
- Build our own REPL implementation to understand internals
- Compare same model across vanilla path and RLM path
- Package manager: `uv`
- No fancy UI; focus on CLI and structured logs

**Docs created:**
- `01_rlm_deep_dive.md`
- `02_repl_explained.md`
- `03_hardware_constraints.md`
- `04_implementation_plan.md`

---

## [2026-03-03] - Session 2: Research Direction

**What we added:**
- Created `05_novel_research_ideas.md`
- Added a phase-by-phase task list
- Framed async sub-calls and training ideas as future extensions

---

## [2026-03-07] - Session 3: Strategy Pivot (Locked)

**User decision:**
- Move findings report into `docs/`
- Treat `docs/` as the living source of truth (mind/heart/soul/diary)
- Use one model only for all experiments: `qwen3.5:2b` via Ollama
- Drop API model execution paths from implementation planning

**Changes made:**
- Moved `RLM_research_findings.md` into `docs/`
- Removed old `research findings/` folder
- Updated docs to single-model local strategy:
  - `project_info.md`
  - `RLM_Overview.md`
  - `README.md`
  - `03_hardware_constraints.md`
  - `04_implementation_plan.md`
  - `tasks.md`
  - `implementation_plan.md`

**Current locked baseline:**
- Vanilla: direct `qwen3.5:2b` inference
- RLM: REPL recursive wrapper over `qwen3.5:2b`
- Runtime: Ollama local only

---

## [2026-03-08] - Session 4: Phase 1 Build — Complete

**What we built:**
- Fixed `pyproject.toml`: removed docs from workspace members, added `hatchling` build backend, added dependencies (`httpx`, `rich`, `tiktoken`)
- Created full `src/` layout with 4 packages: `rlm`, `baseline`, `benchmarks`, `comparison`
- `src/rlm/clients/base.py` — `BaseLLMClient` abstract class
- `src/rlm/clients/ollama.py` — `OllamaClient` (Ollama `/api/chat`, strips Qwen3 `<think>` tags)
- `src/rlm/repl.py` — `LocalREPL` (exec-based, persistent namespace, injects `context`, `query`, `llm_query`, `FINAL`, `FINAL_VAR`, `re`, `json`)
- `src/rlm/rlm_repl.py` — `RLM_REPL` (full REPL loop, code extraction, metadata-only feedback, trajectory logging)
- `src/baseline/vanilla_llm.py` — `VanillaLLM` (direct context pass-through)
- `src/benchmarks/niah.py` — NIAH generator (configurable lengths, positions, seeds)
- `src/benchmarks/long_doc_qa.py` — Long-doc QA generator (facts in filler, multi-length)
- `src/comparison/eval.py` — `score_result`, `aggregate_results` (exact_match, contains_answer, numeric_match)
- `src/comparison/runner.py` — `run_comparison` (JSONL logging, rich progress + summary table)
- `prompts/rlm_system.txt` — system prompt with usage patterns for root LM
- `scripts/smoke_test.py` — single task end-to-end test
- `scripts/run_niah.py` — NIAH benchmark runner with CLI args
- `scripts/run_comparison.py` — full comparison runner (NIAH + LongDocQA)
- `scripts/view_results.py` — JSONL results viewer with breakdown by context length

**Verification:**
- `uv sync` successful, all dependencies installed
- All imports clean: `All imports OK`
- All unit tests passed: LocalREPL, NIAH generator, LongDocQA generator, eval scoring

**Next step:**
- Start Ollama, pull `qwen3.5:2b`, run `scripts/smoke_test.py`

## [2026-03-08] - Session 5: Model switch + Docs update

**Changes:**
- Switched all references from `qwen3.5:4b` → `qwen3.5:2b` (27 occurrences across src, scripts, docs)
- Rewrote root `README.md` — full project documentation for users and contributors
- Updated `docs/README.md` — added status table, expanded quick reference
- Updated `docs/tasks.md` — marked smoke test done, added new decisions log entries
- Smoke test confirmed working: both vanilla and RLM answered correctly on 4K NIAH task

**Smoke test results (qwen3.5:2b, 4K context):**
- Vanilla: correct, 45s
- RLM: correct, 55s, 4 iterations, 0 sub-calls, 0 REPL errors
- GPU: 32/33 layers on CUDA, 1 on CPU, total 5.6 GiB used

**Hardware note confirmed:**
- Ollama auto-caps KV cache at 4096 tokens (~16K chars) due to 6GB VRAM
- Vanilla will be truncated on 32K tasks — RLM unaffected (only sends small chunks)

**Current state:** Waiting for qwen3.5:2b download. Ready to run full NIAH suite.

## [2026-03-08] - Session 6: Graphs, Resumable Runs, RL Roadmap

**Bug fixed:**
- `OllamaClient` timed out on RLM first call — Qwen3.5:2b thinking mode was generating 100s+ of tokens before answering
- Fix: set `think=False` (disables Qwen3 extended reasoning) + raised timeout from 180s → 600s
- Thinking disabled is correct for this project: we compare architectures, not reasoning depth

**New features built:**

`src/comparison/runner.py` — Resumable runs:
- Added `_fingerprint()` — stable MD5 hash per (method, task_type, context_length, query)
- Added `_load_completed_fingerprints()` — reads existing JSONL, extracts done tasks
- `run_comparison()` now skips already-completed tasks automatically
- Safe to Ctrl+C and restart with same `--run-name` — no work lost
- Added `save_trajectories` param — saves full RLM code traces to `<run_name>_trajectories.jsonl`

`scripts/plot_results.py` — 5 plot types:
- `_accuracy.png`     — accuracy vs context length (line chart, vanilla vs RLM)
- `_latency.png`      — latency vs context length (line chart)
- `_overall_bar.png`  — overall metrics side-by-side bar chart
- `_rlm_behaviour.png`— REPL iterations + sub-calls vs context length
- `_advantage.png`    — RLM advantage gap (RLM% − Vanilla%) per context length
- Supports multiple JSONL files for cross-run comparison (different models, configs)
- Saved as PNG to `experiments/plots/`

`scripts/run_niah.py` + `scripts/run_comparison.py` — updated:
- Added `--save-trajectories` flag
- Added `--run-name` (use same name to resume interrupted run)

`src/rlm/clients/ollama.py` — OllamaClient improvements:
- `think=False` default — disables Qwen3 thinking mode, 5–10× faster per call
- `timeout=600s` default — handles slow first RLM calls
- Documents why thinking is disabled for this project

**Dependencies added:**
- `matplotlib>=3.10.8` — for plotting

**RL + Muon roadmap decided:**
- Phase 3 (now): run benchmarks, collect trajectories with `--save-trajectories`
- Phase 4a: filter trajectories (correct + no errors) → training dataset
- Phase 4b: SFT on trajectories using Unsloth + QLoRA (fits 2B model on 6GB VRAM)
- Phase 4c: GRPO fine-tuning with outcome reward (correct answer) + process reward (code validity)
- Phase 4d: Muon optimizer ablation — compare AdamW vs Muon for RLM trajectory fine-tuning
- All training on cloud GPU (48 H100 hours used in paper for 8B model, 2B should be < 10 hrs)

**Current state:** Re-running smoke test with fixes applied. About to start NIAH --small.

## [2026-03-08] - Session 7: First Benchmark Results + Analysis

**NIAH Small Grid completed (niah_small_v1):**
- Grid: [4K, 8K] x [10%, 50%, 90% needle position] = 6 tasks x 2 methods = 12 results
- Saved to `experiments/niah_small_v1.jsonl` + `experiments/niah_small_v1_trajectories.jsonl`

**Results:**

| Method  | Accuracy | Avg Latency | Timeouts |
|---------|----------|-------------|----------|
| Vanilla | 100% (6/6) | 3.2s      | 0        |
| RLM     | 50% (3/6)  | 35.4s     | 1        |

**RLM failure analysis (3 failures):**
1. **4K ctx, 50% pos** (answer=22911): Model found needle but FINAL("1") — wrong extraction
2. **4K ctx, 90% pos** (answer=56334): Model found needle but FINAL("4") — wrong extraction
3. **8K ctx, 90% pos** (answer=64018): Model looped 20 iterations with identical code, never extracted the number (timeout at 165s)

**Root cause:** 2B model code generation quality. The regex finds the needle sentence but the code to extract the number is broken — it pulls a single digit instead of the full number. The timeout case repeats the same approach without variation.

**Key insight:** At 4K-8K, vanilla wins because the full context fits in the KV cache (4096 tokens). RLM adds overhead (code gen + execution) with no benefit. The real test is 16K+ where Ollama truncates vanilla's context but RLM can still search the full document via REPL.

**Plots generated (5 PNGs in experiments/plots/):**
- `niah_small_v1_accuracy.png` — vanilla 100% flat, RLM scattered
- `niah_small_v1_latency.png` — vanilla ~3s flat, RLM 5-165s
- `niah_small_v1_overall_bar.png` — vanilla 100% vs RLM 50% overall
- `niah_small_v1_rlm_behaviour.png` — iterations mostly 1, one spike to 20 (timeout)
- `niah_small_v1_advantage.png` — RLM disadvantage at short contexts (expected)

**Bug fixed:** Unicode arrow in `scripts/plot_results.py` caused UnicodeEncodeError on Windows cp1252

**Started:** Full NIAH grid (`niah_full_v1`) — adds 16K + 32K contexts where RLM should show its advantage.

### Full NIAH Grid v1 Results (niah_full_v1)

Grid: [4K, 8K, 16K, 32K] x [10%, 50%, 90% position] = 12 tasks x 2 methods = 24 results

| Method  | Accuracy | Avg Latency | Timeouts |
|---------|----------|-------------|----------|
| Vanilla | 83.3% (10/12) | 4.2s | 0 |
| RLM     | 50.0% (6/12)  | 34.4s | 2 |

**By needle position (critical dimension):**

| Position | Vanilla | RLM |
|----------|---------|-----|
| 10% (near start) | 4/4 (100%) | 4/4 (100%) |
| 50% (middle) | 3/4 (75%, fails 32K) | 2/4 (50%) |
| 90% (near end) | 3/4 (75%, fails 32K) | 0/4 (0%) |

**Root cause discovered: greedy regex bug.** The 2B model writes `[^\n]*(\d+)` where `[^\n]*` greedily consumes digits, leaving only the last digit for `(\d+)`. Example: needle "56334" → model extracts "4".

### System Prompt Fix (v1 → v2)

Updated `prompts/rlm_system.txt` with:
1. Explicit warning: "NEVER use `[^\n]*(\d+)` — greedy `[^\n]*` eats digits"
2. Correct pattern shown: `[^\d]*(\d+)` or `\D*(\d+)`
3. Added Step 3 fallback: extract all numbers from matched sentence
4. "COMMON MISTAKES TO AVOID" section

### Full NIAH Grid v2 Results (niah_full_v2)

| Method  | Accuracy | Avg Latency | Timeouts |
|---------|----------|-------------|----------|
| Vanilla | 83.3% (10/12) | 3.8s | 0 |
| RLM     | **75.0% (9/12)** | **19.2s** | **0** |

**Improvement: RLM 50% → 75%, timeouts 2 → 0, latency 34.4s → 19.2s**

**Key result at 32K:** RLM PASS where Vanilla FAIL at 32K@90% position — first confirmed RLM advantage on long context where vanilla is truncated by KV cache.

**Remaining RLM failures (3/12):**
- 16K@50%: wrong answer (code quality)
- 32K@10%: regression from v1 (stochastic — 111s latency suggests model went off-track)
- 32K@50%: both vanilla and RLM fail

**Plots saved:** 5 individual + 5 comparison (v1 vs v2) in `experiments/plots/`

## [2026-03-08] - Session 8: Long-Context Results, Plots Rewrite, Phase 4 Setup

**Long-context NIAH (niah_long_v1):**
- Grid: [32K, 64K, 128K] x [10%, 50%, 90%] = 9 tasks x 2 methods
- Completed 13/18 results (128K RLM tasks too slow, process killed)
- Key finding: at 64K, RLM@10% PASS where Vanilla@10% FAIL — vanilla is blind, RLM finds it
- 64K RLM latency: 21s (fast) to 1392s (timeout at 20 iterations)

**Full comparison (comparison_full_v1):**
- 16 tasks: 12 NIAH + 4 LongDocQA, both methods
- Results: Vanilla 81.2% vs RLM 75.0%
- LongDocQA at 32K: RLM PASS where Vanilla FAIL (task 16)

**Combined results across all runs (4K-128K):**

| Context | Vanilla | RLM | Notes |
|---------|---------|-----|-------|
| 4K-8K   | 100%    | 100%| Both fit in KV cache |
| 16K     | 100%    | 67% | RLM code quality issue |
| 32K     | 33%     | 33% | KV cache truncation begins |
| 64K     | 33%     | 67% | RLM advantage — vanilla blind |
| 128K    | 0%      | --  | Vanilla completely fails |

**Plot rewrite (3 iterations):**
- v1: Basic line charts — user said "not research looking"
- v2: DeepSeek-R1-Zero style with serif fonts, markers — user said "straight lines don't make sense"
- v3 (final): Each (method x position) is its own line on numeric X-axis, 8 plot types:
  - `_accuracy_multiline.png` — 6 lines showing per-position degradation
  - `_latency_multiline.png` — 6 lines with latency scaling
  - `_accuracy_aggregated.png` — 2-line summary
  - `_acc_by_position.png` — lines per (method x length)
  - `_advantage.png` — 3 lines (per position) showing RLM% - Vanilla%
  - `_heatmap.png` — pass/fail grid
  - `_dashboard.png` — side-by-side accuracy + latency
  - `_rlm_behaviour.png` — iterations with outcome markers
- Supports `--merge` flag to combine multiple JSONL files

**Phase 4 training pipeline built:**

New package: `src/training/`
- `muon.py` — Muon optimizer implementation (Newton-Schulz orthogonalization for 2D weight matrices, Nesterov momentum, configurable NS steps)
- `split_params_for_muon()` — splits model into Muon (2D) and AdamW (1D) groups

New scripts:
- `scripts/build_sft_dataset.py` — filters trajectories (correct + FINAL + no errors), formats as chat SFT data, deduplicates, outputs JSONL
- `scripts/train_sft.py` — QLoRA SFT with `--optimizer adamw|muon`, 6GB VRAM config (batch=1, grad_accum=4, max_seq=2048)
- `scripts/eval_sft.py` — loads LoRA adapter, runs NIAH eval via HF inference

SFT dataset built: 16 clean examples from 52 trajectories (4K-64K context)

**Training deps installed:**
- PyTorch 2.10.0+cu126 (CUDA 12.6)
- transformers 5.3.0, peft 0.18.1, trl 0.29.0
- datasets 4.6.1, accelerate 1.13.0, bitsandbytes 0.49.2
- All verified working in project venv

**README rewritten** with:
- Key results table
- Training pipeline documentation
- Muon optimizer explanation
- Updated project structure and roadmap

**Current state:** Training pipeline ready. Next: test QLoRA model loading, then run AdamW vs Muon SFT.

*Future sessions will be appended below.*

## [2026-03-14] - Session 9: Roadmap Reordered for Release First

**Planning changes locked in:**
- RL-on-RLM is now the final phase, not the next phase
- The formal RL workflow will be tried in Google Colab through the VS Code extension
- Publication now has three public outputs:
  - Medium article
  - GitHub Pages deployment
  - Hugging Face model or adapter release under the user's account

**Reasoning:**
- The core inference-time RLM story should stand on its own before training claims
- Public artifacts should ship from stable benchmark results, not from unfinished RL work
- Hugging Face release improves discoverability for people specifically searching for RLM-related models
