# RLM Project Research Findings

**Prepared for:** `RLM` project in `docs/`
**Date:** 2026-03-07
**Method:** Internal docs review + external primary-source research (papers, official repos/docs, benchmark dataset cards)

---

## 1) Complete Research Around This Project

### 1.1 What your project is trying to do (validated)
From your internal docs, the goal is clear and strong:
- Build an empirical comparison between vanilla LLM inference and RLM-style inference.
- Focus on long-context degradation ("context rot") and whether recursive decomposition in REPL improves results.
- Stay practical under constrained hardware (RTX 4050 6GB), using API models and/or small local models.

This aligns with the RLM paper direction:
- RLM treats long prompt as external environment state, not direct LM context.
- Root model writes code in REPL, performs sub-calls over slices, and composes final answer.

### 1.2 What external research says (current state)
- **RLM is real and active research**: arXiv preprint (v2 dated 2026-01-28) reports large gains on long-context tasks and introduces post-trained RLM-Qwen3-8B.
- **Official implementation is active**: `alexzhang13/rlm` now has releases (including depth >1 support in latest release title).
- **Benchmark ecosystem matured**:
  - OOLONG (2025) targets aggregation-heavy long-context reasoning.
  - BrowseComp-Plus (2025) introduces fixed corpus for transparent deep-research evaluation.
  - LongBench Pro (2026) reports that effective context length is often below advertised context.
  - BABILong (2024) also reports strong degradation and low effective context utilization.

### 1.3 Implementation landscape
- **Official `rlm` repo**: most complete, supports multiple environments and model providers.
- **`rlm-minimal`**: useful for educational baseline and architecture clarity.
- **Community `rlm_repl`**: useful engineering ideas (cost tracking, configurable endpoint, timeout behavior) but not a substitute for rigorous evaluation harness.

---

## 2) What You Are Doing Wrong (According to Your Docs) and How to Prevent It

### 2.1 Unfair baseline design
**Problem in docs:** `docs/04_implementation_plan.md` says vanilla baseline should be deliberately unfair (no chunking/RAG).  
**Why this is wrong:** It weakens scientific validity. External readers will reject conclusions as benchmark gaming.  
**Prevention:** Compare at least 4 baselines:
1. Vanilla direct context (current baseline)
2. Vanilla + best-effort retrieval baseline (BM25 or dense retrieval)
3. CodeAct/ReAct tool-use baseline
4. RLM

Keep model family and token budget comparable.

### 2.2 Over-reliance on synthetic/simple tasks
**Problem in docs:** NIAH gets heavy attention; real-task coverage is thin.  
**Why this is wrong:** NIAH is necessary but not sufficient; it does not capture aggregation-intensive reasoning.  
**Prevention:** Use mixed benchmark suite:
- NIAH (smoke test)
- OOLONG / OOLONG-Pairs style aggregation tasks
- BrowseComp-Plus style fixed-corpus deep-research tasks
- LongBench/LongBench Pro style realistic multitask long-context evaluations

### 2.3 Security underestimation in local `exec` REPL
**Problem in docs:** Local REPL is treated as generally fine with limited caution.  
**Why this is wrong:** Python docs explicitly warn `exec` executes arbitrary code; prompt injection + tool invocation can become host compromise risk.  
**Prevention:**
- Default to isolated sandbox (Docker or remote sandbox) for anything except local toy tests.
- Restrict tool permissions and runtime capabilities.
- Add input/output filtering, audit logs, and rate limits.

### 2.4 Outdated assumptions on recursion depth and roadmap
**Problem in docs:** Depth>1 is framed mostly as future work.  
**Why this is incomplete:** Official repo release metadata now references depth >1 support (Feb 18, 2026).  
**Prevention:** Update docs to reflect current upstream capabilities; position depth experiments as near-term optional ablations.

### 2.5 RL plan is exciting but currently underspecified
**Problem in docs:** RL is discussed conceptually, but no explicit reward model design, environment schema, or data-quality gates.  
**Why this is risky:** RL on tool-using trajectories is sensitive to reward hacking and formatting failures.  
**Prevention:** Define concrete RL protocol (see Section 4 below), including process constraints and trajectory validity checks.

### 2.6 Too many unstable assumptions hard-locked early
**Problem in docs:** Model/provider/cost choices are "LOCKED IN" too early.  
**Why this is risky:** API model names, limits, and pricing change frequently.  
**Prevention:** Treat provider/model as config, not strategy. Benchmark harness must be provider-agnostic.

---

## 3) What Should Change to Make This Succeed

### 3.1 Change the evaluation contract first
Define one strict experiment spec:
- Same task set and seeds for all methods.
- Same max wall-clock and max token budget per query.
- Report accuracy/F1 + cost + latency + call counts.
- Run at least 3 seeds and report mean/std.

### 3.2 Add trajectory-level observability
For each run, log:
- Root iterations
- Sub-call count
- Recursion depth
- REPL error types
- FINAL/FINAL_VAR correctness
- Token/cost split: root vs sub-calls

This converts "it works" into diagnosable science.

### 3.3 Introduce safety guardrails as default
- Disable dangerous builtins/modules by default in REPL profile.
- Add execution timeout per step and global budget per sample.
- Add max file/network/tool policy.
- Add prompt-injection-aware separation of instructions and untrusted context.

### 3.4 Re-scope stretch goals
Priority order to maximize publishable output:
1. Reproducible LLM vs RLM comparison (core)
2. Async sub-call runtime optimization (engineering paper/blog quality)
3. RL fine-tuning pilot on small model with strict rewards (research extension)
4. Muon optimizer ablation (only after stable RL/SFT baseline)

### 3.5 Build reproducibility package from day one
Ship:
- `configs/` for every experiment
- frozen dataset version ids / commit hashes
- deterministic seeds
- one-command evaluation script
- report generator (`md` + table exports)

---

## 4) RL on the RLM Idea (Feasibility + Recommendation)

### 4.1 Is RL on RLM a good idea?
**Yes, but only after a strong supervised baseline.**  
RLM behavior is process-heavy (tool-use, code generation, decomposition), so process-aware RL is a reasonable next step.

### 4.2 Why RL can help here
RLM quality depends on intermediate decisions:
- chunking strategy
- when to recurse
- how many sub-calls
- when to terminate
- whether to preserve/compose intermediate state correctly

These are exactly the kinds of behaviors process rewards can optimize.

### 4.3 Practical RL recipe for your project
Stage-gated approach:
1. **SFT stage** on high-quality RLM trajectories (clean, executable, correct FINAL usage).
2. **Reward stage** with mixed reward:
   - Outcome reward: answer correctness
   - Process reward: valid code, fewer failures, budget-aware behavior
   - Penalty: unsafe tool usage, runaway loops, malformed finalization
3. **RL stage** with async-capable trainer/environment stack.

Recommended tooling direction (from current ecosystem):
- Environment-centric RL stack (`verifiers`, `prime-rl`) is now available for LLM RL workflows.
- Process-reward literature (PRIME, implicit PRM work) gives a useful blueprint for sparse-label settings.

### 4.4 What not to do in RL phase
- Do not start RL before your non-RL harness is stable.
- Do not optimize only for final accuracy; include cost and safety penalties.
- Do not trust unfiltered trajectories; enforce strict trajectory validators.

### 4.5 Compute realism
- RTX 4050 6GB: good for harness development, light local inference, and debugging.
- Meaningful RL/SFT experiments on 7B+ generally require cloud GPU.
- Paper-level RLM post-training example used only 1,000 samples but still reported **48 H100 hours**.

---

## 5) What You Should Take Care Of

- **Evaluation integrity:** no cherry-picked datasets or hand-tuned prompts per method.
- **Prompt leakage/data contamination:** avoid exposing benchmark answers in prompts/logs.
- **Cost accounting:** split root/sub-call costs and include retries/timeouts.
- **Failure-mode tracking:** syntax errors, dead loops, bad FINAL usage, hallucinated variable names.
- **Security posture:** sandbox + least privilege + monitoring must be default once external inputs are involved.

---

## 6) Things to Keep in Mind

- RLM is a **systems paradigm**, not just a prompt trick.
- Improvements come from orchestration quality as much as model quality.
- Claims of "unbounded context" are practical only with good decomposition and budget management.
- Benchmark diversity matters more than single leaderboard gains.
- Provider-specific limits change frequently; keep architecture provider-agnostic.

---

## 7) Revealing Facts Found During Research

1. **Official RLM repo appears to have moved faster than your docs:** release metadata references depth>1 support and history compaction (latest release title dated 2026-02-18).
2. **RLM paper reports strong gains but also clear limitations:** synchronous sub-calls and shallow recursion acknowledged as constraints.
3. **Training data quality is a major bottleneck:** paper reports non-trivial FINAL/FINAL_VAR error rates in collected trajectories before filtering.
4. **Benchmarking landscape now emphasizes fairness and transparency:** BrowseComp-Plus specifically addresses reproducibility problems of live web benchmarks.
5. **"Long context" marketing is not equal to effective reasoning length:** LongBench Pro and BABILong both report substantial effective-length gaps.

---

## Action Plan (Recommended Next 14 Days)

1. Rewrite evaluation spec and baseline matrix in docs.
2. Implement logging schema and failure taxonomy before running big experiments.
3. Add sandbox profile and security controls for REPL execution.
4. Run first reproducible comparison on a mixed benchmark subset.
5. Decide RL pilot only after baseline report is stable.

---

## Sources (Primary)

### Project-internal docs reviewed
- `docs/project_info.md`
- `docs/RLM_Overview.md`
- `docs/01_rlm_deep_dive.md`
- `docs/02_repl_explained.md`
- `docs/03_hardware_constraints.md`
- `docs/04_implementation_plan.md`
- `docs/05_novel_research_ideas.md`
- `docs/tasks.md`

### External references
- RLM paper (v2): https://arxiv.org/abs/2512.24601
- RLM PDF (results/limitations details): https://arxiv.org/pdf/2512.24601
- Official RLM repo: https://github.com/alexzhang13/rlm
- RLM minimal repo: https://github.com/alexzhang13/rlm-minimal
- Community repo: https://github.com/fullstackwebdev/rlm_repl
- OOLONG benchmark: https://arxiv.org/abs/2511.02817
- BrowseComp-Plus benchmark: https://arxiv.org/abs/2508.06600
- BrowseComp-Plus dataset card (stats): https://huggingface.co/datasets/Tevatron/browsecomp-plus
- LongBench Pro: https://arxiv.org/abs/2601.02872
- LongBench: https://arxiv.org/abs/2308.14508
- BABILong: https://arxiv.org/abs/2406.10149
- PRIME (process RL): https://arxiv.org/abs/2502.01456
- Free Process Rewards without labels: https://arxiv.org/abs/2412.01981
- Prime Intellect verifiers (RL envs): https://github.com/PrimeIntellect-ai/verifiers
- Prime-RL: https://github.com/PrimeIntellect-ai/prime-rl
- Python `exec` docs: https://docs.python.org/3/library/functions.html#exec
- OWASP LLM prompt injection cheat sheet: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html
