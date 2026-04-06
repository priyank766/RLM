# How RLMs Work — A Step-by-Step Walkthrough

> The idea is simple: give the model a programming environment instead of a text box. The details are where it gets interesting.

---

## The Big Picture

Here's the simplest way to think about it. A standard LLM is like a person who can only read — you hand them a document, they read it, they answer questions about it. An RLM is like a person who can read **and** use a computer — you hand them the document as a file, they write scripts to search it, extract from it, summarize sections, and piece together an answer.

Both approaches use the same underlying intelligence. But one of them has tools.

---

## The Architecture, Layer by Layer

### Layer 1: The REPL Environment

The REPL (Read-Eval-Print Loop) is a persistent Python environment. Think of it like a Jupyter notebook that runs automatically — code cells get executed one at a time, and variables persist between cells.

When an RLM run starts, the REPL is initialized with several things injected into its namespace:

```python
namespace = {
    "context": <the full document as a string>,   # The long prompt — in RAM, not in the LLM
    "query": <the user's question>,               # What we want answered
    "context_len": 52719,                         # Convenience: document length

    # Helper functions the model can call:
    "head(n)":         first n characters of context
    "tail(n)":         last n characters of context
    "context_slice()": arbitrary character range
    "chunk_text()":    split context into overlapping chunks
    "keyword_windows()":  find occurrences of a keyword with surrounding text
    "regex_windows()":    find regex matches with surrounding text
    "query_chunks()":     batch-process chunks through sub-LM calls

    # The recursive call function — this is the magic:
    "llm_query(prompt)":   call the sub-LM on an arbitrary prompt

    # Termination signals:
    "FINAL(answer)":   call with the final answer string
    "FINAL_VAR(var)":  call to return a variable's value as the answer
}
```

Critically, `context` is a Python string in memory. It is **not** in the LLM's context window. The model can access any part of it through the helper functions, but it never sees the whole thing at once.

### Layer 2: The Root LLM

The root LLM is the orchestrator. It's the same model you'd use for vanilla inference — in my case, `qwen3.5:2b`. But its prompt looks very different:

```
System: You are an RLM. Write Python code to examine the context
variable and answer queries. Use helper functions like
keyword_windows(), llm_query(), and chunk_text().
For long contexts, gather evidence first, then verify with
at least one focused sub-call before FINAL.

User:
Context info:
  Total length : 52,719 characters
  Preview      : It is of the first importance to not allow yourself...

Query: What was Holmes trying to figure out?

Write Python code to examine the context and answer the query.
Call FINAL(answer) when you have the answer.
```

That's the entire prompt. A few lines of metadata. The model generates Python code as its response.

### Layer 3: The Loop

Here's what happens next, step by step:

**Step 1: Root LLM generates code**

The model writes a Python code cell. Something like:

```python
# Let me start by understanding the context
print(f"Document length: {context_len} characters")
print("First 500 characters:")
print(head(500))
```

**Step 2: REPL executes the code**

The code runs in the persistent namespace. Output:

```
Document length: 52719 characters
First 500 characters:
It is of the first importance to not allow yourself to be influenced by any
preconceived theory. Lay the facts before the mind, and then reason from them...
```

**Step 3: Output metadata goes back to the root LLM**

Only a truncated version of the output is appended to the LLM's conversation history:

```
[stdout | 623 chars]
Document length: 52719 characters
First 500 characters:
It is of the first importance to not allow yourself...
```

The model sees this and generates the next code cell.

**Step 4: The model searches for relevant sections**

```python
# Search for the main character names and key events
import re
# Find character names (capitalized words in context)
names = keyword_windows("Holmes", window=400, limit=3)
for i, window in enumerate(names):
    print(f"--- Holmes mention {i+1} ---")
    print(window[:300])
```

**Step 5: The model recurses on focused snippets**

Now it gets interesting. The model takes one of those keyword windows and asks itself a question about it:

```python
# Ask about the first Holmes mention
result = llm_query("""
Based on this text:
--- Holmes mention 1 ---
"...Mr. Sherlock Holmes, who is usually late..."

Question: What problem or case is Holmes working on in this passage?
Answer concisely — return ONLY the answer, no explanation.
""")
print(f"Sub-LM answer: {result}")
```

This `llm_query()` call is a **separate invocation of the same model**. But the prompt is tiny — just a 400-character snippet and a focused question. No dilution. No distractors.

**Step 6: The model consolidates and terminates**

After gathering evidence from a few sub-calls, the model has enough information:

```python
FINAL("Holmes was trying to recover a compromising photograph from Irene Adler, who threatened to send it to his royal client's fiancée.")
```

The REPL detects the `FINAL()` call, captures the answer, and the loop terminates.

---

## The Recursive Sub-Call Mechanism

This is the part that makes RLMs genuinely powerful, so let me dwell on it a bit.

When the root LLM calls `llm_query()`, here's what actually happens:

1. The `llm_query()` function in the REPL namespace receives the prompt string
2. It constructs a messages list: `[{"role": "user", "content": prompt}]`
3. It calls the **sub-LM client** (which, in my setup, is the same `qwen3.5:2b` model)
4. The sub-LM returns an answer
5. The answer is auto-printed in the REPL output (so the root LLM sees it) and returned

The sub-LM call is completely isolated. It has no memory of previous iterations. It gets a clean, focused prompt and returns a clean, focused answer. This is fundamentally different from how a vanilla LLM handles long documents — where all the information is crammed into one prompt and the model has to figure out what's relevant on its own.

In my experiment on the 52,719-character Sherlock Holmes document, the model answered the query in exactly 3 sub-calls. Each sub-call was a focused question about a specific ~400-character snippet. The root LLM then synthesized those answers into a final response.

---

## The Recursive-First Policy

One thing I added to my implementation that the paper alludes to but doesn't enforce is a **recursive-first policy** for long contexts. Here's the problem: the model, being lazy, will often try to answer immediately from the preview text without doing any real exploration. It'll call `FINAL()` on the first iteration with whatever it can guess from the first 300 characters.

For long documents, this is usually wrong.

So I built in a rejection mechanism. When the context is longer than 16,000 characters:

1. **First-iteration FINAL() is always rejected.** The model must do at least some exploration first.
2. **FINAL() without any sub-calls is rejected.** The model must have asked itself at least one focused question.
3. **Repeated code strategies are flagged.** If the model writes nearly identical code two iterations in a row, the feedback pushes it to try a different approach.

After two rejections, the model is allowed to finalize anyway — it's a nudge, not a hard constraint. But in practice, the nudge works. The model almost always does better on its second or third attempt once it's been forced to gather evidence.

---

## What the REPL Actually Protects

A concern with any `exec()`-based system is: what if the model redefines the helper functions? What if it writes `def keyword_windows(...)` with a broken version?

I added protection for this. The REPL maintains a list of protected names — all the helper functions, the context variable, the `FINAL` function. Before each code execution, it snapshots their current values. After execution, it checks if any of them were overwritten and restores the originals if so. It also strips out function definitions that shadow built-in helpers before the code even runs.

The model gets a warning in its stderr output:

```
[REPL WARNING] You redefined built-in function(s): keyword_windows.
This was blocked. These functions are ALREADY available — just call them directly.
Example: results = keyword_windows('search term')
```

---

## How Termination Works

There are two ways the RLM loop ends:

### Clean termination: FINAL() or FINAL_VAR()

The model calls one of these functions, which sets a flag in the REPL namespace. After each code execution, the loop checks this flag. If set, it returns the answer.

### Timeout

If the model hits `max_iterations` (default: 20) without calling FINAL, the loop returns `None`. This isn't a failure — it's an honest signal that the model couldn't converge. Forcing a bad answer is worse than returning nothing.

In my experiments, timeouts happened occasionally with the 2B model, especially on harder tasks. They're useful diagnostic signals: a timeout means the model's code generation quality isn't sufficient for the task, which is itself an interesting data point.

---

## The Trajectory

Every RLM run produces a trajectory — a step-by-step record of what the model wrote, what the REPL output was, whether it errored, and whether it tried to finalize. Here's what one looks like (abbreviated):

```json
{
  "iteration": 0,
  "code": "# Let me start by understanding the context\nprint(head(500))",
  "stdout": "It is of the first importance to not allow yourself...",
  "stderr": "",
  "final_set": false
}
{
  "iteration": 1,
  "code": "windows = keyword_windows(\"Holmes\", limit=3)\nfor i, w in enumerate(windows):\n    print(f\"Window {i}: {w[:200]}\")",
  "stdout": "Window 0: ...Mr. Sherlock Holmes, who is usually late...\nWindow 1: ...Holmes whistled.\nWindow 2: ...\"Not a bit, Doctor. Stay where you are...",
  "stderr": "",
  "final_set": false
}
{
  "iteration": 2,
  "code": "result = llm_query(\"Based on this text... What is Holmes trying to figure out?\")\nprint(f\"Answer: {result}\")",
  "stdout": "[Sub-LM answered] Holmes is trying to recover a compromising photograph.\nAnswer: Holmes is trying to recover a compromising photograph.",
  "stderr": "",
  "final_set": false
}
{
  "iteration": 3,
  "code": "FINAL(\"Holmes was trying to recover a compromising photograph from Irene Adler.\")",
  "stdout": "",
  "stderr": "",
  "final_set": true
}
```

These trajectories are useful beyond just debugging. They're training data. Each correct trajectory — one that reaches the right answer with clean code — is an example of how a model *should* approach a long-context task. I'm collecting these for supervised fine-tuning, which is the next phase of this project.

---

## Why This Is Different From Agents

You might be thinking: isn't this just another coding agent? Isn't this what Devin does, or what Cursor does?

The answer is no, and the distinction matters.

Agentic systems have the LLM's context window accumulate everything — all the tool outputs, all the previous observations, all the intermediate results. As the agent works longer, its context fills up. It eventually suffers from the same context rot it was designed to avoid.

An RLM is structurally different. The root model's context window **never grows beyond a fixed bound**. The long document never enters it. The REPL outputs are truncated before they're appended. The only things that accumulate are short code cells and their truncated outputs. The model can process an arbitrarily large document without its context window ever getting large.

This isn't an incremental improvement. It's a different way of thinking about how an LLM interacts with information.

---

## The Code

If you want to see the actual implementation, the key files are:

- **`src/rlm/repl.py`** — The LocalREPL class. exec-based, persistent namespace, helper functions, protection guards.
- **`src/rlm/rlm_repl.py`** — The RLM_REPL class. The main loop, code extraction, metadata feedback, recursive-first policy.
- **`src/rlm/clients/ollama.py`** — The OllamaClient. HTTP calls to the local Ollama server, thinking mode disabled for speed.
- **`src/baseline/vanilla_llm.py`** — The VanillaLLM class. Direct context inference, no REPL, for comparison.

The whole thing runs on a laptop. No cloud, no API, no special hardware. Just Python and Ollama.

---

*Next: see the actual experiment I ran on a 52,719-character document, including the real trajectory and what happened at each iteration.*
