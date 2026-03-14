import re
import time
from pathlib import Path
from typing import Any

from .clients.base import BaseLLMClient
from .repl import LocalREPL

# How many chars of REPL stdout to feed back to the root LM per iteration.
# KEY: never put full output in the LM context window.
_MAX_FEEDBACK_CHARS = 600

# Truncate sub-LM responses to keep REPL namespace lean.
_MAX_SUB_RESPONSE_CHARS = 1200


class RLM_REPL:
    """
    Recursive Language Model with REPL execution.

    Wraps any BaseLLMClient to provide RLM-style inference:
      - context is stored in an external REPL (never in LLM context window)
      - root LM writes Python code cells iteratively
      - llm_query() enables recursive sub-calls on focused chunks
      - session ends on FINAL() / FINAL_VAR() or max_iterations

    Both root_client and sub_client can be the same model instance
    (which is the default — fair single-model comparison).
    """

    def __init__(
        self,
        root_client: BaseLLMClient,
        sub_client: BaseLLMClient | None = None,
        max_iterations: int = 20,
        system_prompt: str | None = None,
    ):
        self.root_client = root_client
        self.sub_client = sub_client or root_client
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or _load_system_prompt()
        self._repl = LocalREPL()

    def completion(self, context: str, query: str) -> dict[str, Any]:
        """
        Run RLM inference on context + query.

        Returns a dict with:
            answer      — str | None (None on timeout)
            iterations  — number of REPL turns used
            sub_calls   — number of llm_query() invocations
            repl_errors — list of stderr snippets from failed cells
            timed_out   — bool
            latency_s   — wall-clock seconds
            trajectory  — list of per-iteration records (code, stdout, stderr)
        """
        t0 = time.time()
        sub_call_count = 0
        repl_errors: list[str] = []
        trajectory: list[dict] = []

        def llm_query(prompt: str) -> str:
            nonlocal sub_call_count
            sub_call_count += 1
            msgs = [{"role": "user", "content": prompt}]
            response = self.sub_client.complete(msgs)
            # Truncate to keep REPL namespace lean
            if len(response) > _MAX_SUB_RESPONSE_CHARS:
                return response[:_MAX_SUB_RESPONSE_CHARS] + "... [truncated]"
            return response

        self._repl.setup(context, query, llm_query)

        # Build initial message: give root LM ONLY metadata, not the full context
        preview = context[:300].replace("\n", " ")
        initial_user = (
            f"Context info:\n"
            f"  Total length : {len(context):,} characters\n"
            f"  Preview      : {preview}...\n\n"
            f"Query: {query}\n\n"
            f"Write Python code to examine the context and answer the query. "
            f"Call FINAL(answer) when you have the answer."
        )

        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user},
        ]

        for iteration in range(self.max_iterations):
            # Root LM generates the next code cell
            raw_response = self.root_client.complete(history)
            code = _extract_code(raw_response)

            # Execute code in REPL
            stdout, stderr = self._repl.execute(code)

            trajectory.append({
                "iteration": iteration,
                "code": code,
                "stdout": stdout,
                "stderr": stderr,
                "final_set": self._repl.is_final,
            })

            if stderr:
                repl_errors.append(stderr[:300])

            # Check if LLM called FINAL()
            if self._repl.is_final:
                return {
                    "answer": self._repl.final_answer,
                    "iterations": iteration + 1,
                    "sub_calls": sub_call_count,
                    "repl_errors": repl_errors,
                    "timed_out": False,
                    "latency_s": round(time.time() - t0, 2),
                    "trajectory": trajectory,
                }

            # Append code + short output metadata back to history
            consecutive_errors = sum(
                1 for e in trajectory[-3:] if e.get("stderr")
            ) if len(trajectory) >= 1 else 0
            history.append({"role": "assistant", "content": raw_response})
            history.append({"role": "user", "content": _format_feedback(stdout, stderr, consecutive_errors)})

        # Reached max_iterations without a FINAL call
        return {
            "answer": None,
            "iterations": self.max_iterations,
            "sub_calls": sub_call_count,
            "repl_errors": repl_errors,
            "timed_out": True,
            "latency_s": round(time.time() - t0, 2),
            "trajectory": trajectory,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_code(response: str) -> str:
    """
    Extract Python code from an LLM response.

    Priority:
      1. ```python ... ``` fenced block
      2. ``` ... ``` fenced block
      3. Raw response as fallback
    """
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def _format_feedback(stdout: str, stderr: str, consecutive_errors: int = 0) -> str:
    """
    Format REPL output for the root LM.

    Only sends a short preview + length — never the full output.
    This is the key mechanism that prevents context window pollution.

    When there are consecutive errors, escalates the recovery instruction
    so the model understands it must change its approach.
    """
    parts: list[str] = []

    if stdout:
        truncated = len(stdout) > _MAX_FEEDBACK_CHARS
        preview = stdout[:_MAX_FEEDBACK_CHARS]
        label = f"[stdout | {len(stdout)} chars" + (" | truncated]" if truncated else "]")
        parts.append(f"{label}\n{preview}")

    if stderr:
        # Extract just the error type and message — strip the long traceback
        error_lines = stderr.strip().splitlines()
        short_error = "\n".join(error_lines[-3:])  # last 3 lines = the actual error
        parts.append(
            f"[REPL ERROR — your code threw an exception]\n"
            f"{short_error}\n\n"
            f"Your previous code DID NOT WORK. You MUST write completely different code."
        )
        if consecutive_errors >= 2:
            parts.append(
                f"[WARNING: {consecutive_errors} consecutive errors]\n"
                f"STOP using the same approach. Try a much simpler strategy:\n"
                f"  1. Print context[:500] to see what the document looks like\n"
                f"  2. Use re.search() to find the answer directly\n"
                f"  3. Do NOT reference variables that you haven't defined in this same code block"
            )
    elif not stdout:
        parts.append("[no output — code ran silently, no errors]")

    parts.append("Write new Python code. Call FINAL(answer) when you have the answer.")
    return "\n\n".join(parts)


def _load_system_prompt() -> str:
    """Load system prompt from prompts/rlm_system.txt, or use built-in default."""
    # src/rlm/rlm_repl.py -> src/rlm/ -> src/ -> project root
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "rlm_system.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM_PROMPT


_FALLBACK_SYSTEM_PROMPT = """\
You are an RLM (Recursive Language Model). You answer queries about long documents \
using a Python REPL.

The full document is in the variable `context`. You cannot see it directly — \
write Python code to examine it.

AVAILABLE IN REPL:
  context     (str)  — full document
  query       (str)  — question to answer
  context_len (int)  — length of context in characters
  llm_query(prompt: str) -> str  — call a sub-LM on a focused prompt
  FINAL(answer: str)             — submit final answer (ends session)
  FINAL_VAR(var)                 — submit a variable as final answer
  re, json   — pre-imported

RULES:
1. Write ONLY Python code. No prose outside code comments.
2. Use context[start:end] to read slices.
3. Keep llm_query() chunks under 4000 characters.
4. Always end with FINAL() when you have the answer.
5. Variables persist across iterations — build incrementally.
6. To extract numbers: use [^\\d]*(\\d+) NOT [^\\n]*(\\d+) — greedy [^\\n]* eats digits!\
"""
