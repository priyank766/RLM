# RLM Deep Dive: Complete Research Notes
> Source: arxiv:2512.24601 (Zhang, Kraska, Khattab), Alex Zhang blog, alexzhang13/rlm repo, alexzhang13/rlm-minimal, fullstackwebdev/rlm_repl

---

## 1. The Problem: Why LLMs Fail at Long Context

### Context Rot
- As context grows longer, LLM inference quality **degrades steeply** — even frontier models.
- This is called **"context rot"**: the model gets "dumber" the more context you pile into it.
- Unlike simple needle-in-haystack tests (which modern models handle fine at 1M+ tokens),
  **information-dense tasks** suffer degradation at even moderate lengths (8K–64K tokens).
- Anthropic defines it as: "The number of tokens in context window increases → ability to recall info decreases."
- Reality is worse: it's not retrieval failure but reasoning failure. Dense tasks like OOLONG-Pairs require
  reasoning over almost every line — current models get F1 < 0.1% without RLM.

### Existing Solutions (and why they suck)
| Approach | What it does | Why it fails |
|---|---|---|
| Context compaction / summarization | Summarize old context when limit hit | Lossy — details needed later get lost |
| RAG / retrieval agents | Retrieve snippets into context | Still bounded by context window; heuristic retrieval |
| Coding agents (ReAct) | LLM reads external data, executes code | External data only; verbalize sub-calls autoregressively; no true recursion |
| Sub-agent delegation | LLM invokes itself as sub-agent | Bounded by output length; no programmatic looping |

**Key insight from paper**: All prior approaches have the prompt P inside the LLM context window.
They are fundamentally bounded. RLMs are the first approach to treat P as an **external object**.

---

## 2. What is an RLM? (The Core Idea)

A **Recursive Language Model (RLM)** is an inference-time wrapper around a base LLM (ℳ) that:
1. **Never feeds the full context P directly into ℳ's context window.**
2. Instead, stores P as a **variable in an external REPL (Python) environment**.
3. Allows ℳ to write Python code to peek, filter, chunk, and process P programmatically.
4. Allows code running in the REPL to **call back into ℳ** (or a smaller sub-LM) on sub-slices of P.

From the user's perspective it is a **drop-in replacement**:
```python
# Vanilla LLM:
result = llm.completion(messages)

# RLM (same interface, unbounded context):
result = rlm.completion(messages)
```

Under the hood it's totally different.

---

## 3. RLM Architecture — How It Actually Works

### The Algorithm (simplified from Algorithm 1 in the paper)

```
Given: prompt P (can be arbitrarily long)

1. Initialize REPL environment with:
   - Variable `context` = P  (stored in memory, NOT in LLM context)
   - Function `llm_query(prompt)` = calls sub-LM on arbitrary prompt

2. Build initial LLM history with ONLY metadata:
   - Length of P
   - Short prefix of P  
   - Instructions on how to use REPL

3. LOOP:
   a. Call root LLM ℳ with current history → ℳ generates Python code
   b. Execute code in REPL → output captured
   c. Append ONLY SHORT METADATA of output to LLM history (NOT full output)
      → This is the KEY: forces ℳ to use variables, never pollute its window
   d. If REPL sets Final variable → STOP, return Final as response
   e. If max_iterations reached → return None (timeout)
```

### Three Design Points That Make RLMs Different

| Point | RLM (Algorithm 1) | Naive Agentic LLM (Algorithm 2) |
|---|---|---|
| **Where P lives** | External REPL variable (never in LLM context) | Directly in LLM context window |
| **Output generation** | Via REPL variables (unbounded) | Autoregressively (bounded by context window) |
| **Sub-calls** | Programmatic — inside loops, over O(\|P\|) or O(\|P\|²) slices | Verbalized one-at-a-time, explicit delegation |

### Key Mechanics

**How the LLM terminates:**
```python
FINAL("my answer here")       # Direct string answer
FINAL_VAR(my_var)             # Return value from a REPL variable
```

**How the LLM calls sub-LMs:**
```python
# Inside REPL, the LLM writes:
result = llm_query("Summarize this: " + context[0:5000])
results = [llm_query(f"Process chunk {i}: {chunk}") for i, chunk in enumerate(chunks)]
```

---

## 4. What is a REPL?

**REPL = Read-Eval-Print Loop**

Think of it like a persistent Python terminal session where the LLM writes code line by line and sees the outputs.

Like a **Jupyter Notebook** cell-by-cell execution — but non-interactive, controlled by the LLM.

### How it works in RLMs:
1. **Read**: LLM generates a Python code cell
2. **Eval**: Python `exec()` runs the code in a persistent namespace
3. **Print**: stdout/stderr captured and returned
4. **Loop**: Process repeats until `Final` is set or max iterations hit

### What's available inside the REPL:
- The full context P as `context` variable (in memory as a Python string)
- Standard Python libraries (re, json, etc.)
- The `llm_query(prompt)` function for recursive LLM sub-calls
- Persistent state — all variables from previous iterations are available

### Implementation: Local vs Isolated REPL

| Type | Mechanism | Use Case | Security |
|---|---|---|---|
| **Local** | Python `exec()` in same process | Development, benchmarking | Not safe for prod |
| **Docker** | `DockerREPL` in a container | Semi-isolated dev | Medium |
| **Modal Sandboxes** | Cloud-isolated VMs | Production | High |
| **Prime Intellect** | Sandboxed remote machines | Production | High |
| **e2b / Daytona** | Cloud sandboxes | Production | High |

**For our project**: Local REPL (exec-based) is perfectly fine for research/comparison purposes.

---

## 5. Sub-Agents / Recursive Calls — How Recursion Works

The power of RLMs comes from **programmatic recursion** — not just calling another LLM once, but calling it inside loops.

### Depth-1 Recursion (What the paper uses)
```
Root LM (depth=0) → orchestrates, decomposes, writes code
  └── Sub-LM (depth=1, often smaller) → answers questions about sub-chunks
```

- Root LM sees ONLY metadata about P and short REPL outputs
- Sub-LM gets focused, small slices of P — fits in context window easily
- Sub-LM can be **the same model** or a cheaper/smaller one

### Depth-N Recursion (Future work)
```
Root LM (depth=0)
  └── Sub-LM (depth=1, itself an RLM)
        └── Sub-Sub-LM (depth=2)
```
Enabling this is simple in code: replace `Sub_RLM` with `RLM_REPL` class.
But it wasn't needed for existing benchmarks (depth=1 was sufficient).

### Emergent Behaviors Observed in Paper

The authors didn't train the LLMs to be RLMs — they just prompted them. Even without training, emergent
strategies emerged:

1. **Chunking + recursive sub-calls**: Split context by newlines, loop over chunks, call sub-LM on each.
2. **Filter/search first**: Use regex to find relevant sections, then recurse only on those.
3. **Stitch outputs via variables**: Sub-LM outputs stored in list/dict, later 
   combined into final answer. Enables unbounded output length.

---

## 6. Key Results from the Paper

### Benchmarks Used
| Benchmark | Task Type | Context Length | Complexity |
|---|---|---|---|
| OOLONG | Long-context QA (1 hop) | 8K–256K tokens | Linear |
| OOLONG-Pairs | Long-context QA (2 hop) | 8K–256K tokens | Quadratic |
| BrowseComp-Plus | Deep research task | 1M–11M tokens | Linear |
| CodeQA | Code understanding | 64K–256K tokens | Linear |
| S-NIAH | Needle in haystack | 8K–256K tokens | Constant |

### RLM vs Vanilla LLM Results
| Benchmark | Vanilla GPT-5 | RLM(GPT-5) | Improvement |
|---|---|---|---|
| OOLONG | Baseline | +28.4% | Significant |
| OOLONG-Pairs F1 | <0.1% | 58.0% | **Massive — emergence** |
| BrowseComp-Plus | Baseline | +29% over retrieval baseline | Best method |

**Post-trained RLM**: RLM-Qwen3-8B (post-trained on RLM trajectories) outperforms base Qwen3-8B by **28.3% average** and approaches GPT-5 performance on 3 tasks.

### Key Tradeoffs
- **Short contexts**: Base LLM slightly beats RLM (overhead of REPL loop)
- **Context >16K tokens**: RLM consistently wins
- **Cost**: RLM average cost ≈ comparable or cheaper (smarter token usage via filtering)
- **REPL alone helps**: Even without sub-calls, just REPL context management outperforms RAG

---

## 7. Negative Results (Paper Appendix B) — Things That Didn't Work

This is important — the authors were transparent:
- **Direct prompting to stay within small chunks**: Models would often "cheat" by putting answers in root context
- **Compaction within REPL**: Tried summarizing context in REPL before sub-calling — hurt performance
- **DeepSearch-style retrieval inside REPL**: Keyword-only retrieval was worse than recursive sub-calls
- **Very deep recursion (depth > 1)**: Didn't help for current benchmarks; adds complexity

---

## 8. Existing Implementations

### alexzhang13/rlm (Official, Production-Grade)
- **PyPI package**: `pip install rlms`
- **Use with `uv`**: `uv init && uv pip install -e .`
- REPL environments: local, docker, modal, prime, daytona, e2b
- Model providers: OpenAI, Anthropic, OpenRouter, Portkey, LiteLLM, vLLM
- Built-in trajectory logger + JSON visualizer
- Clean class hierarchy: `RLM` → `RLM_REPL` → specific env classes

### alexzhang13/rlm-minimal (Simple Gist-Style)
- Stripped-down, educational implementation
- `rlm/rlm_repl.py`: Main `RLM_REPL` class with `completion()` method
- `rlm/repl.py`: exec-based REPL with `llm_query()` injected
- Example: NIAH with ~1M lines of random words
- Dependencies: openai, python-dotenv, rich

### fullstackwebdev/rlm_repl (Community Implementation)
- Based on the paper + rlm-minimal
- Supports configurable API endpoint (`RLM_API_URL` env var) → works with llama.cpp, vLLM, etc.
- Cost tracking for root and sub-LLM separately
- max_iterations and max_output_length configurable
- Returns `None` on timeout (respects paper's "natural convergence" philosophy)
- Supports local LLM servers (e.g. llama.cpp with Qwen3-Coder quantized models)

---

## 9. Summary Timeline

| Date | Event |
|---|---|
| Dec 2025 | arXiv preprint v1 released |
| Oct 2025 | Original blog post by Alex Zhang |
| 2026 (ongoing) | arXiv v2 updated; official `rlms` PyPI package; multiple community implementations |
