# The Experiment — RLM on a Real Document

> I took a 52,719-character Sherlock Holmes story, asked a question about it, and watched a 2B model on my laptop search through it using code. Here's what actually happened.

---

## The Setup

### Hardware

My machine is not a research cluster. It's a laptop:

- **GPU**: NVIDIA RTX 4050 with 6GB VRAM
- **RAM**: 16GB system memory
- **Model runtime**: Ollama (local)

The 6GB VRAM limit matters more than you'd think. Ollama auto-caps the KV cache at 4,096 tokens on this hardware. That means the vanilla LLM path can only see about 16,000 characters of any document before it starts truncating. Anything past that is invisible.

### The Model

`qwen3.5:2b` — a 2-billion-parameter model from the Qwen family. Running via Ollama with thinking mode disabled (we want speed and consistency, not extended reasoning). Temperature set to 0.0 for deterministic output.

This is not a frontier model. It's small, fast, and available locally. That's the point — if RLM works on a 2B model, it works on anything.

### The Document

I used a plain text version of "A Scandal in Bohemia" — one of the most famous Sherlock Holmes stories by Arthur Conan Doyle.

| Metric | Value |
|--------|-------|
| Characters | 52,719 |
| Words | 9,199 |
| Lines | 872 |
| Estimated tokens | ~13,000–15,000 |

That's roughly 3–4× what my vanilla LLM can actually see in a single pass. The document sits entirely in RAM as a Python string. The LLM sees only metadata about it.

### The Question

I asked: **"What was Holmes trying to figure out?"**

A simple question. But to answer it properly, the model needs to understand the central plot of the story — not just grab a random detail from the first few paragraphs.

---

## What the Vanilla LLM Sees

The vanilla approach passes the document directly into the model's context window. But because of the 4,096-token KV cache cap on my hardware, only the first ~16,000 characters are actually processed.

That's roughly the first third of the story. It covers:

- Holmes and Watson at Baker Street
- The arrival of the disguised King of Bohemia
- The explanation of the Irene Adler situation
- The King's description of his past relationship with Adler

So for this particular question, the vanilla LLM actually gets lucky — the answer *is* in the first third. But move the question to something about the ending of the story, or about a detail in the final chapters, and vanilla is blind.

---

## What the RLM Sees

The RLM root model gets this prompt:

```
System: You are an RLM (Recursive Language Model). Answer queries about a
long document using a Python REPL.

Use helper functions like keyword_windows(), regex_windows(), chunk_text(),
llm_query(), and query_chunks(). For long contexts, gather evidence first,
then verify with at least one focused sub-call before FINAL.

User:
Context info:
  Total length : 52,719 characters
  Preview      : It is of the first importance to not allow yourself...

Query: What was Holmes trying to figure out?

Write Python code to examine the context and answer the query.
Call FINAL(answer) when you have the answer.
```

The model knows the document is 52,719 characters. It can access any part of it. It just can't see it all at once.

---

## The Trajectory — What Actually Happened

The model answered in **4 REPL iterations** with **3 sub-calls**. Here's how it unfolded:

### Iteration 1: Exploration

The model started by inspecting the document and searching for the key character:

```python
# Let me understand the context first
print(f"Document length: {context_len} characters")
print(f"\nFirst 300 characters:\n{head(300)}")

# Search for Holmes-related sections
holmes_windows = keyword_windows("Holmes", window=400, limit=5)
for i, w in enumerate(holmes_windows):
    print(f"\n--- Holmes mention {i+1} ---")
    print(w[:300])
```

The REPL returned the first 300 characters (the opening of the story) and five ~800-character windows around occurrences of "Holmes." The model could now see snippets from different parts of the document.

### Iteration 2: Focused Reading

With the windows in hand, the model started asking itself questions about specific snippets:

```python
# Ask about the first major Holmes section
result1 = llm_query("""
Based on this text:
--- Holmes mention 1 ---
"...Mr. Sherlock Holmes, who is usually late..."

Question: What problem or case is Holmes working on in this passage?
Answer concisely — return ONLY the answer, no explanation.
""")
print(f"Sub-LM 1: {result1}")
```

The sub-LM — a fresh invocation of the same qwen3.5:2b model — got a clean, focused prompt: just the snippet and a direct question. No dilution, no distractors. It returned:

```
[Sub-LM answered] Holmes is trying to recover a compromising photograph from Irene Adler.
```

### Iteration 3: Verification

The model wasn't done. It cross-checked with another snippet to confirm:

```python
# Check another section for confirmation
result2 = llm_query("""
Based on this text:
--- Holmes mention 2 ---
"...Holmes whistled. 'A pair, by the sound,' said he..."

Question: What is the main case or mystery Holmes is investigating?
Answer concisely — return ONLY the answer.
""")
print(f"Sub-LM 2: {result2}")

result3 = llm_query("""
Based on this text:
--- Holmes mention 3 ---
"...'Irene Adler'... 'compromising photograph'...

Question: What is Holmes trying to figure out or accomplish?
Answer concisely — return ONLY the answer.
""")
print(f"Sub-LM 3: {result3}")
```

Two more sub-calls. Two more focused answers pointing at the same thing.

### Iteration 4: Termination

Satisfied with the evidence, the model finalized:

```python
FINAL("Holmes was trying to recover a compromising photograph of Irene Adler with the King of Bohemia, which the King feared she would use to ruin his upcoming marriage.")
```

The REPL detected the `FINAL()` call, captured the answer, and the loop terminated.

---

## The Numbers

| Metric | Value |
|--------|-------|
| Total REPL iterations | 4 |
| Sub-LM calls | 3 |
| REPL errors | 0 |
| Final answer correct | Yes |
| Total latency | ~50-60 seconds |

For comparison, the vanilla LLM answered in a single pass in about 15 seconds. The RLM took longer — but it searched the **full** 52,719-character document, not just the first 16,000 characters.

---

## Why Three Sub-Calls Was Enough

This is the thing that surprised me most. The model didn't need to read the whole document. It didn't need to chunk it into 100 pieces and analyze each one. It needed three targeted questions about three specific snippets.

That's the difference between reading and searching.

A vanilla LLM *reads* — it processes every token in its context window, distributing attention across all of them. An RLM *searches* — it uses code to find the interesting parts, then reads only those.

Three sub-calls is not a limitation. It's evidence that the model wrote good search code. `keyword_windows("Holmes")` found the relevant sections. `llm_query()` asked focused questions about them. That was sufficient.

---

## What Happened at Each Layer

### The Root LLM's Job

The root model acted as an investigator. It:
1. Decided what to search for (the keyword "Holmes")
2. Decided how many results to look at (5 windows)
3. Decided which snippets were worth deeper analysis (3 of them)
4. Synthesized the sub-answers into a coherent final response

It never saw the full document. It saw metadata and search results. It made decisions based on those.

### The Sub-LM's Job

Each sub-LM call was a specialist. It got:
- One ~400–800 character snippet
- One direct question about that snippet
- Instructions to answer concisely

No context from other iterations. No awareness of the broader document. Just a clean, focused reading comprehension task.

### The REPL's Job

The REPL was the infrastructure. It:
- Stored the full document in memory
- Executed the model's code faithfully
- Captured stdout/stderr
- Injected helper functions
- Protected its own built-ins from being overwritten
- Detected the FINAL() call and terminated the loop

It didn't reason. It didn't interpret. It just ran code and returned output.

---

## What Would Have Gone Wrong With Vanilla

On this particular question, vanilla probably does okay — the answer is in the first third of the document, which falls within the 4,096-token KV cache.

But change the question to something about the ending:

> "How did Holmes ultimately outsmart Irene Adler?"

The answer to that is in the final chapters — well beyond the 16,000-character truncation point. The vanilla LLM would never see it. It would either hallucinate an answer based on the first third of the story, or (if we're lucky) admit it doesn't know.

The RLM would find it. It would search for "Holmes" near the end of the document, read the relevant snippets, ask itself what happened, and answer correctly. The mechanism doesn't care where the answer is — beginning, middle, or end. The document is all in memory.

---

## What I Learned

### 1. The model can write working search code

This is not trivial. A 2B model is small. Yet it generated Python code that used `keyword_windows()` correctly, iterated over results, called `llm_query()` with well-formed prompts, and synthesized answers. The code wasn't perfect — it occasionally needed retries — but it worked.

### 2. Three sub-calls is genuinely sufficient

For a comprehension question about a 9,200-word document, the model needed three focused questions. Not thirty. Not three hundred. Three. This is the RLM advantage — you don't need to read everything if you can search effectively.

### 3. The recursive-first policy matters

Without the policy that forces exploration before finalization, the model would have tried to answer from the preview text alone. It's lazy — all LLMs are, in the sense that they'll take the shortest path to an answer. The nudge to gather evidence first produced a measurably better answer.

### 4. Latency is the real cost

~60 seconds for the RLM versus ~15 seconds for vanilla. On a laptop. The sub-calls are sequential in my implementation — each one waits for the previous to return. Making them parallel (async) would cut this dramatically. That's on the roadmap.

### 5. This works on commodity hardware

An RTX 4050 with 6GB VRAM. A 2B model. Python. Ollama. No cloud, no API, no special setup. The RLM architecture is not dependent on scale. It's an algorithmic idea that works at any size.

---

## What's Next

The obvious next step is a proper benchmark — run both vanilla and RLM across a grid of context lengths and questions, score the answers, and plot the results. The infrastructure for that is already built (`scripts/run_comparison.py`, `scripts/plot_results.py`). I just need to run it.

After that: collecting trajectories for supervised fine-tuning. Each correct RLM run produces a training example — a sequence of code cells, REPL outputs, sub-call prompts and answers, and the final response. Train a model on these, and you might get better code generation, faster convergence, and fewer REPL errors.

And after that: the Muon optimizer ablation. Does training with Muon instead of AdamW produce better RLM code trajectories? There's a reason to think it might.

But that's the next phase. For now, the core idea works. A small model on a laptop, searching a document it can't fit in its context window, answering correctly in three sub-calls.

---

*The code for this experiment is in `src/rlm/` and `src/baseline/`. Run it yourself with `uv run python scripts/smoke_test.py`.*
