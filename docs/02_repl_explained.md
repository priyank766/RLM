# REPL — What It Is, How It Works, and How We'll Build Ours

---

## 1. What is a REPL?

**REPL = Read-Eval-Print Loop**

It's the programming model behind interactive Python shells (like you get when you type `python` in a terminal, or run cells in a Jupyter notebook).

```
Read   → Accept input (code or text)
Eval   → Execute it
Print  → Show the output
Loop   → Repeat
```

A REPL maintains **persistent state** across interactions — variables defined in step 1 are available in step 2, 3, etc. This is unlike running isolated scripts.

### Interactive Python REPL (example):
```python
>>> x = 10
>>> y = 20
>>> x + y
30
>>> items = [1, 2, 3]
>>> sum(items)
6
```

All variables (`x`, `y`, `items`) persist between cell evaluations.

---

## 2. REPL in the Context of RLMs

In RLMs, the LLM **plays the role of the human** typing into the REPL. Each LLM generation becomes a "cell" that gets executed.

### The loop looks like:
```
LLM generates Python code (cell)
    ↓
Python exec() runs the code
    ↓
stdout/stderr captured
    ↓
Metadata about output fed back to LLM
    ↓
LLM generates next cell
    ↓
... repeat until Final is set
```

### What makes it special:
- The long context P is stored as `context` variable in REPL — **never in LLM context window**
- The REPL injects a special `llm_query(prompt)` function the LLM can call recursively
- State persists — the LLM builds up partial answers across multiple turns

### Example trajectory (what the LLM might actually write across iterations):

**Iteration 1:**
```python
# First, understand the context
print(f"Context length: {len(context)} chars")
print(f"First 500 chars: {context[:500]}")
```

**Iteration 2:**
```python
# Search for relevant keyword
import re
matches = [(m.start(), context[max(0,m.start()-100):m.start()+200]) 
           for m in re.finditer(r'La Union', context)]
print(f"Found {len(matches)} matches")
```

**Iteration 3:**
```python
# Recursively ask sub-LM about each match
answers = []
for idx, (pos, chunk) in enumerate(matches):
    ans = llm_query(f"Based on this text: {chunk}\n\nAnswer: What festival was celebrated in La Union?")
    answers.append(ans)
print(answers)
```

**Iteration 4:**
```python
# Consolidate and finish
final_answer = answers[0]  # or merge logic
FINAL(final_answer)
```

---

## 3. How to Build a REPL (Python exec-based)

The simplest possible REPL for RLMs uses Python's **`exec()`** built-in.

```python
class SimpleREPL:
    def __init__(self, context: str, llm_client):
        # Persistent namespace — all variables live here
        self._namespace = {
            "context": context,                        # The long context P
            "llm_query": self._make_llm_query(llm_client),  # Recursive call function
            "__builtins__": __builtins__,
        }
        self._stdout_history = []

    def _make_llm_query(self, llm_client):
        """Create the llm_query() function that lives inside REPL."""
        def llm_query(prompt: str) -> str:
            """Sub-LM call: sends prompt to a sub-LM, returns answer string."""
            # Build messages for sub-LM call
            messages = [{"role": "user", "content": prompt}]
            response = llm_client.complete(messages)
            return response.text
        return llm_query

    def execute(self, code: str) -> str:
        """Run one cell. Returns captured stdout."""
        import io, contextlib
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            try:
                exec(code, self._namespace)
            except Exception as e:
                return f"ERROR: {e}"
        output = stdout_capture.getvalue()
        self._stdout_history.append(output)
        return output

    def check_final(self):
        """Check if LLM set a FINAL variable."""
        return self._namespace.get("__final__", None)
```

### The FINAL() convenience function (injected into REPL namespace):
```python
def FINAL(answer: str):
    """Called by LLM to signal completion."""
    # Store in a special namespace variable
    frame = sys._getframe(1)
    frame.f_locals["__final__"] = answer
    # Actually we inject this differently — see full impl

def FINAL_VAR(var):
    """Signals that 'var' (a REPL variable) contains the answer."""
    frame = sys._getframe(1)
    frame.f_locals["__final__"] = str(var)
```

---

## 4. LLM Loop — How the Root LM Drives the REPL

```python
class RLM_REPL:
    def __init__(self, root_model, sub_model, repl_env, max_iterations=20):
        self.root_model = root_model
        self.sub_model = sub_model
        self.repl = repl_env
        self.max_iterations = max_iterations

    def completion(self, context: str, query: str) -> str:
        # Initialize REPL with context
        self.repl.init(context)

        # Initial history: ONLY metadata, not the context itself
        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"""
Context info:
- Total length: {len(context)} characters  
- First 200 chars: {context[:200]}...

Query: {query}

Use the REPL to interact with the full context stored in `context` variable.
Call llm_query() for recursive analysis. Use FINAL() when done.
"""}
        ]

        for iteration in range(self.max_iterations):
            # Root LM generates code
            response = self.root_model.complete(history)
            code = response.text

            # Execute in REPL
            output = self.repl.execute(code)
            
            # Check for termination
            final = self.repl.check_final()
            if final:
                return final

            # Append ONLY SHORT metadata of output back to history
            # KEY: never put full output into LLM context
            history.append({"role": "assistant", "content": code})
            history.append({"role": "user", "content": f"[Output len: {len(output)}] {output[:300]}..."})

        return None  # Timeout
```

---

## 5. Types of REPL Environments — Tradeoffs

### LocalREPL (exec-based) — What We'll Use
```python
exec(code, namespace)
```
- **Pros**: Simple, zero config, fast, uses same Python environment
- **Cons**: No isolation — LLM-generated code runs with full host permissions
- **For us**: Fine for research/benchmarking — we control the queries

### DockerREPL
- Runs code in a Docker container
- The `context` variable and `llm_query` function are serialized/passed in
- **Pros**: Isolated, safe
- **Cons**: Needs Docker installed, slower startup

### Cloud REPLs (Modal, Prime, e2b)
- Runs code on remote VMs
- LLM sub-calls reach back to host machine
- **Pros**: Most secure, scalable
- **Cons**: Needs API keys, network latency, money

---

## 6. What to Inject Into the REPL Namespace

Essential:
```python
namespace = {
    "context": P,                    # The raw context string
    "llm_query": sub_lm_call,        # Recursive sub-LM call
    "FINAL": set_final_answer,       # Termination signal
    "FINAL_VAR": set_final_from_var, # Termination via variable
    "__builtins__": __builtins__,    # Standard Python
}
```

Optional but helpful:
```python
namespace.update({
    "re": re,       # Regex for searching
    "json": json,   # JSON parsing
    "context_len": len(P),  # Convenience
})
```

---

## 7. Practical Considerations for Our Build

| Consideration | Decision |
|---|---|
| REPL type | Local exec-based (simplest) |
| Sub-LM | Same model as root (for fair comparison) or cheaper model |
| max_iterations | 20 (paper default) |
| max_output_length | 500,000 chars per REPL cell before truncation |
| Timeout behavior | Return `None` (don't force bad answer) |
| Output metadata | First 300 chars + total length in root LLM history |
| FINAL detection | Check REPL namespace after each exec() |
