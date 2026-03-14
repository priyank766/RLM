# Project Tasks & Status

> This is the master task tracker. Update it every session.
> Legend: `[ ]` = todo, `[/]` = in progress, `[x]` = done

---

## Phase 0: Research & Planning
- [x] Read RLM paper (arXiv:2512.24601)
- [x] Read Alex Zhang blog post
- [x] Analyze `alexzhang13/rlm` repo (official)
- [x] Analyze `alexzhang13/rlm-minimal` repo
- [x] Analyze `fullstackwebdev/rlm_repl` repo
- [x] Create docs directory structure
- [x] Write `01_rlm_deep_dive.md` - comprehensive paper breakdown
- [x] Write `02_repl_explained.md` - REPL concept + build guide
- [x] Write `03_hardware_constraints.md` - hardware reality check
- [x] Write `04_implementation_plan.md` - full project plan
- [x] Add `RLM_research_findings.md` inside `docs/`
- [x] Finalize single-model strategy: Ollama `qwen3.5:2b` only

## Phase 1: Core Infrastructure
- [x] Initialize `uv` project (`pyproject.toml`, `.venv`)
- [x] Write `src/rlm/clients/base.py` - `BaseLLMClient` abstract class
- [x] Write `src/rlm/clients/ollama.py` - Ollama local client for `qwen3.5:2b`
- [x] Write `src/rlm/repl.py` - `LocalREPL` class (exec-based)
- [x] Persistent namespace with `context`, `llm_query`, `FINAL`, `FINAL_VAR`
- [x] stdout/stderr capture per cell
- [x] Final answer detection
- [x] Write `src/rlm/rlm_repl.py` - `RLM_REPL` class
- [x] `completion(context, query)` runs REPL loop
- [x] Pass only metadata (not full output) back to root LM
- [x] max_iterations and timeout handling
- [x] Write `src/baseline/vanilla_llm.py` - `VanillaLLM` wrapper
- [x] `completion(context, query)` for direct same-model baseline
- [x] Write system prompts (`prompts/rlm_system.txt`)
- [x] NIAH smoke test - passed on `qwen3.5:2b` (4K context, both methods correct)

## Phase 2: Benchmarks
- [x] Write `src/benchmarks/niah.py` - NIAH generator
- [x] Configurable context lengths
- [x] Random word haystack + number needle
- [x] Write `src/benchmarks/long_doc_qa.py` - long document QA
- [x] Generated QA pairs with known answers
- [x] Write `src/comparison/runner.py` - comparison harness
- [x] Runs both vanilla and RLM on same benchmark
- [x] Logs to JSONL in `experiments/`
- [x] Write `src/comparison/eval.py` - scoring
- [x] Exact match / contains answer / numeric match
- [x] Latency tracking

## Phase 2.5: Infrastructure Additions
- [x] Resumable runs - skip already-done tasks on restart (fingerprint-based)
- [x] `--save-trajectories` flag - saves full RLM code traces to JSONL for later training
- [x] `scripts/plot_results.py` - multiple plot types
- [x] Multi-run comparison in plots
- [x] `OllamaClient`: `think=False` default, timeout raised to 600s
- [x] Fixed Qwen3 read timeout issue in thinking mode
- [x] Rewrote `prompts/rlm_system.txt` to be recursive-first on long contexts
- [x] Added REPL helper functions: `head`, `tail`, `context_slice`, `chunk_text`, `keyword_windows`, `regex_windows`, `query_chunks`
- [x] Added long-context runtime guard: reject premature `FINAL()` and require at least one focused sub-call before finalization

## Phase 3: Analysis
- [x] Run NIAH experiments across small, full, and long grids
- [x] Run mixed benchmark comparison (`NIAH` + `LongDocQA`)
- [x] Diagnose major RLM failures and patch prompt issues
- [x] Re-run experiments after prompt fixes
- [x] Generate comparison plots
- [x] First confirmed RLM > Vanilla result at long context
- [ ] Document observed emergent RLM behaviors from trajectories
- [ ] Write analysis summary: where RLM wins, where it struggles, latency tradeoff
- [/] Collect 200+ successful trajectories for later RL training

## Phase 4: RL-on-RLM
- [ ] Use Google Colab through the VS Code extension for the training workflow
- [ ] Build the filtered SFT dataset from clean trajectories
- [ ] Run SFT baseline and compare against the base model
- [ ] Define reward function for RL / GRPO phase
- [ ] Set up RL training loop
- [ ] Train RL-on-RLM and evaluate vs vanilla, base RLM, and SFT-RLM
- [ ] Compare reward curves, accuracy, and REPL error rates
- [ ] If useful, test Muon vs AdamW inside this Colab-based training phase

## Phase 5: Publication and Public Release
- [ ] Write Medium article: research-style write-up of system, experiments, and findings
- [ ] Build GitHub Pages deployment with plots, methodology, and benchmark results
- [ ] Prepare Hugging Face release plan for the public RLM artifact
- [ ] Publish a Hugging Face model or adapter under your account with clear naming and model card so people searching for RLM can find it
- [ ] Align README, docs, and public artifacts around the same benchmark story

## RL Prep Already Explored Early
- [x] Write `src/training/muon.py` - Muon optimizer
- [x] Write `scripts/build_sft_dataset.py` - build SFT data from trajectories
- [x] Write `scripts/train_sft.py` - QLoRA SFT script
- [x] Write `scripts/eval_sft.py` - evaluate trained adapters
- [x] Build initial SFT dataset from collected trajectories
- [x] Install training dependencies in the project environment
- [ ] Keep these as preparation only until formal Phase 4 begins

---

## Decisions Log

| Date | Decision | Reasoning |
|---|---|---|
| 2026-03-03 | Build REPL from scratch | Max learning, full control |
| 2026-03-03 | No fancy UI | Research project, CLI is enough |
| 2026-03-03 | `uv` package manager | User preference |
| 2026-03-07 | Move research findings into `docs/` | Keep docs as single source of truth |
| 2026-03-07 | Use only Ollama (no API models) | Fair comparison and reproducibility |
| 2026-03-08 | Switch from `qwen3.5:4b` to `qwen3.5:2b` | Hardware fit - 2b runs faster on 6GB VRAM |
| 2026-03-08 | Metadata-only REPL feedback | Prevents root context pollution |
| 2026-03-08 | Save trajectories separately from results | Keeps results JSONL clean |
| 2026-03-14 | RL moves before publication | We want to test training improvements before locking the public story |
| 2026-03-14 | RL training workflow will use Colab + VS Code extension | Better fit for cloud training while staying in the same workspace flow |
| 2026-03-14 | Public release includes Hugging Face | Lets others discover and try the RLM artifact directly |
| 2026-03-14 | Long contexts use recursive-first policy | Prevents one-shot regex answers from dominating RLM behavior |
| 2026-03-14 | REPL exposes chunking and evidence helpers | Makes sub-calls and multi-step exploration easier for the root model |



