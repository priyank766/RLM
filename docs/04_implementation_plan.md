# Project: LLM vs RLM Comparison
## Full Implementation Plan

---

## Goal

Build a clean, reproducible comparison framework that shows:
1. What RLM does better than vanilla LLM on long contexts
2. Where RLM is worse or overkill
3. How recursive REPL behavior changes reasoning quality

This is a research project, not a product UI.

---

## Locked Model Strategy

- **Only one model is used for all paths:** `qwen3.5:2b`
- **Runtime:** Ollama (local only)
- **No API providers** in the execution plan
- **Fairness:** vanilla and RLM both use the exact same base model

---

## Architecture Overview

```
Comparison Runner
  - loads benchmark tasks (context + query)
  - runs Vanilla path (same model)
  - runs RLM path (same model + REPL loop)
  - collects metrics

Vanilla LLM path:
  qwen3.5:2b -> direct context + query

RLM path:
  qwen3.5:2b (root)
    -> REPL executes generated Python
    -> llm_query() invokes qwen3.5:2b on sub-chunks
    -> FINAL()/FINAL_VAR() returns answer
```

---

## Project Structure (Planned)

```
RLM/
|-- docs/
|-- src/
|   |-- rlm/
|   |   |-- __init__.py
|   |   |-- repl.py
|   |   |-- rlm_repl.py
|   |   `-- clients/
|   |       |-- base.py
|   |       `-- ollama.py
|   |-- baseline/
|   |   |-- __init__.py
|   |   `-- vanilla_llm.py
|   |-- benchmarks/
|   |   |-- __init__.py
|   |   |-- niah.py
|   |   `-- long_doc_qa.py
|   `-- comparison/
|       |-- __init__.py
|       |-- runner.py
|       `-- eval.py
|-- experiments/
|-- prompts/
|-- pyproject.toml
`-- README.md
```

---

## Phase-by-Phase Plan

### Phase 1: Core Infrastructure
- [ ] Set up `uv` project and dependencies
- [ ] Write `BaseLLMClient` with `complete(messages) -> str`
- [ ] Implement `OllamaClient` for `qwen3.5:2b`
- [ ] Write `LocalREPL` class (persistent namespace + stdout/stderr capture)
- [ ] Inject `context`, `llm_query`, `FINAL`, `FINAL_VAR` into REPL
- [ ] Write `RLM_REPL` completion loop
- [ ] Write `VanillaLLM` wrapper
- [ ] Basic NIAH smoke test

### Phase 2: Benchmarks and Logging
- [ ] Build benchmark runner for `(context, query, ground_truth)`
- [ ] Implement NIAH generator
- [ ] Implement long document QA benchmark
- [ ] Track: exact match, token estimates, latency, REPL iterations, sub-call count
- [ ] Log results as JSONL in `experiments/`

### Phase 3: Analysis
- [ ] Run context-length sweeps (small to large)
- [ ] Compare accuracy, latency, and behavior traces
- [ ] Identify where RLM starts helping
- [ ] Document observed failure modes

### Phase 4: Publication and Packaging
- [ ] Write the main research-style Medium article
- [ ] Prepare GitHub Pages deployment with plots, methodology, and benchmark summaries
- [ ] Package and publish a public Hugging Face model/checkpoint under the project identity so others can discover the RLM work

### Phase 5: RL-on-RLM Final Phase
- [ ] Move RL work to the final phase only, after core results and public write-up are stable
- [ ] Run RL / SFT experiments in Google Colab through the VS Code extension workflow
- [ ] Evaluate whether RL materially improves recursive code generation, convergence, and long-context accuracy
- [ ] Keep Muon and related optimizer ablations inside this final cloud-based training phase

---

## Key Design Decisions

### Same model for root, sub-calls, and baseline
- Vanilla: direct inference with `qwen3.5:2b`
- RLM: recursive inference with `qwen3.5:2b`
- Isolates architecture difference from model-quality differences

### Structured logging
- One run file per experiment in `experiments/`
- Entry schema includes method, context length, answer, correctness, token estimates, latency, and recursion metadata

---

## Success Criteria

| Metric | Target |
|---|---|
| RLM accuracy on long-context tasks | Measurably above vanilla on selected tasks |
| Vanilla vs RLM latency profile | Clearly documented trade-off |
| Reproducibility | Same setup reruns with stable outcomes |
| Documentation quality | All decisions and outcomes captured in `docs/` |
| Publication readiness | Results are clear enough to publish on Medium, GitHub Pages, and Hugging Face |

---

## Technologies and Libraries

| Library | Purpose | Install |
|---|---|---|
| `httpx` | Ollama HTTP client | `uv add httpx` |
| `rich` | terminal output and reports | `uv add rich` |
| `datasets` | benchmark dataset loading (optional) | `uv add datasets` |
| `tiktoken` | token estimation | `uv add tiktoken` |

---

## Open Questions

1. Does `qwen3.5:2b` reliably generate executable REPL code?
2. What prompt design minimizes invalid code and looping?
3. What is the practical context-size crossover where RLM starts winning?
4. How should we enforce safe REPL execution for local experiments?
5. Which benchmark mix gives fair and realistic conclusions?
6. What exact Hugging Face artifact do we want to publish at the end: adapter, full checkpoint, or both?
