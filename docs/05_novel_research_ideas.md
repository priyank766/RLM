# Novel Research Ideas: Extending RLMs
> Can we build something genuinely new? Yes. Here's an honest analysis of gaps and opportunities.

---

## Framing: What "Original Research" Means Here

The RLM paper (Zhang, Kraska, Khattab, 2026) is very recent. It explicitly lists open research directions.
This is a **massive opportunity** - the field is young, under-explored, and low-hanging fruit exists.

We are not chasing a Nobel Prize. We're looking for **meaningful, implementable contributions**
that could realistically become a blog post, technical report, a public model release, or even an arXiv preprint.

**Be honest about scope**: We cannot pretrain a 70B model. We *can* do experiments on small models,
contribute a new training recipe, or develop a novel architectural idea and test it at small scale.

---

## Category 1: Training RLMs with Better Optimizers (The Muon Angle)

### What Muon Is
Muon (Momentum + Orthogonalization via Newton-Schulz iterations) was introduced in late 2024.
It applies **orthogonalized gradient updates** to 2D weight matrices (hidden layer parameters).

| Property | AdamW | Muon |
|---|---|---|
| Works on | All parameters | 2D matrix params (hidden layers) |
| Scalar/embed params | Yes | Still uses AdamW for these |
| Training speed | Baseline | **35% faster** |
| Token efficiency | Baseline | **~15% fewer tokens** needed |
| Cost reduction | Baseline | ~50% cheaper training |
| Memory | Baseline | ~33% reduction |
| Real-world use | Standard | Kimi K2 (2T params), NanoGPT SOTA |

### The Connection to RLMs

The RLM paper post-trained `RLM-Qwen3-8B` on RLM trajectories (context -> REPL code steps -> final answer).
This training was done with standard supervised fine-tuning plus likely AdamW.

**The Research Question**:
> *Does replacing AdamW with Muon (or Muon+AdamW hybrid) for post-training a natively recursive model
> result in faster convergence, better generalization, or more efficient REPL trajectory generation?*

### Why This Is Interesting

1. **RLM training data is sparse and expensive** - you only get a few thousand high-quality trajectories.
2. **RLM training involves lots of matrix operations** - REPL code generation is heavy on 2D weight matrices.
3. **This hasn't been tried** - Muon + RLM remains a realistic experimental angle.

### Realistic Experiment

**Setup:**
```
Base model: small code-capable model
Training data: 500-1000 filtered RLM trajectories
Experiment A: Fine-tune with AdamW
Experiment B: Fine-tune with Muon + AdamW split
Metric: NIAH accuracy, REPL code validity, convergence speed
```

**Hardware reality**:
- Generate trajectories locally
- Train in Google Colab when needed
- Keep inference and evaluation local where possible

---

## Category 2: Asynchronous Parallel Sub-Calls (Architecture Contribution)

### The Problem
Current RLMs use **synchronous sub-calls**: the REPL pauses and waits for each `llm_query()` to return.

### The Idea: Async Sub-Call REPL

Replace synchronous blocking sub-calls with async parallel sub-calls.

**Expected improvement**: 10-100x latency reduction for information-dense tasks.

### What We'd Need to Build
1. `AsyncLocalREPL`
2. `async_llm_query()`
3. Async-aware `RLM_REPL.completion()`
4. Prompting that teaches the model to use `await` and `asyncio.gather()`

### Why This Is Publishable
- Simple idea, significant impact, not in the paper
- Clear benchmarking story: same accuracy, much faster on parallel tasks

---

## Category 3: RLM with Adaptive Recursion Depth

### The Problem
Current RLMs usually use shallow recursion.

### The Idea
Let the model decide when deeper recursion is worth the cost.

### Why This Is Interesting
- Could improve hierarchical reasoning
- Naturally connects to RL-based policy decisions

---

## Category 4: REPL Code Quality as a Trained Skill

### The Honest Observation
A major failure mode is still poor Python generation by the root model.

### The Idea
Use process-aware training or rewards to improve:
1. Code that executes cleanly
2. Code that uses REPL state correctly
3. Code that converges efficiently
4. Code that avoids unnecessary recursive overhead

---

## Category 5: Structured Output from RLM REPL

### The Idea
Extend RLM outputs beyond final strings into JSON, tables, or extraction-friendly artifacts.

This is useful for document intelligence and public demos, but it is secondary to the core benchmark story.

---

## Feasibility Matrix

| Idea | Hardware Needed | Difficulty | Novelty | Our Priority |
|---|---|---|---|---|
| Async parallel sub-calls | RTX 4050 sufficient | Low | High | **#1** |
| Structured output REPL | RTX 4050 sufficient | Low | Medium | **#2** |
| Adaptive recursion depth | Cloud GPU | High | Very High | #3 |
| Muon for RLM post-training | Colab / cloud GPU | Medium | High | **Final phase** |
| PRM / RL for code quality | Colab / cloud GPU | Very High | Very High | **Final phase** |

---

## Honest Assessment: What We Should Publish

**Primary public outputs:**
1. Medium article documenting the system, experiments, and lessons
2. GitHub Pages deployment with plots, methodology, and benchmark results
3. Public Hugging Face release of the final RLM-related model artifact so people searching for RLM work can find it

**Optional advanced output:**
4. Technical report or arXiv-style write-up if async or RL results are strong enough

---

## Suggested Research Path

```
Phase 1: Build and stabilize the comparison framework
  -> Understand the system deeply and collect clean results

Phase 2: Run benchmark suite and trajectory collection
  -> Establish where RLM wins and where it fails

Phase 3: Write the analysis and prepare public artifacts
  -> Medium article + GitHub Pages + Hugging Face release plan

Phase 4: Publish the core project publicly
  -> Share results before training claims get ahead of the evidence

Phase 5: RL-on-RLM final phase
  -> Use Google Colab through the VS Code extension workflow
  -> Run SFT / RL experiments and evaluate whether training actually helps
```

---

## Practical Note on RL Phase

RL should stay last for this project.
- The base system and benchmark story need to be stable first.
- Publication of the core inference-time RLM work should not depend on a successful RL outcome.
- Colab + VS Code extension is the right place to try the training phase because it keeps local setup lighter while still letting us work from the same codebase.

---

## Sources for This Section

- Muon optimizer: https://github.com/KellerJordan/modded-nanogpt
- RLM limitations: arXiv:2512.24601 Appendix B + Section 6
- Async future work: explicitly stated in RLM paper Section 6
