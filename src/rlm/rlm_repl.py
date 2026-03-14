import difflib
import re
import time
from pathlib import Path
from typing import Any

from .clients.base import BaseLLMClient
from .repl import LocalREPL

_MAX_FEEDBACK_CHARS = 600
_MAX_SUB_RESPONSE_CHARS = 1200
_LONG_CONTEXT_THRESHOLD = 16_000
_MAX_PREMATURE_FINAL_REJECTIONS = 2
_REPEAT_SIMILARITY_THRESHOLD = 0.92


class RLM_REPL:
    """Recursive Language Model with REPL execution."""

    def __init__(
        self,
        root_client: BaseLLMClient,
        sub_client: BaseLLMClient | None = None,
        max_iterations: int = 20,
        system_prompt: str | None = None,
        long_context_threshold: int = _LONG_CONTEXT_THRESHOLD,
    ):
        self.root_client = root_client
        self.sub_client = sub_client or root_client
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or _load_system_prompt()
        self.long_context_threshold = long_context_threshold
        self._repl = LocalREPL()

    def completion(self, context: str, query: str) -> dict[str, Any]:
        t0 = time.time()
        sub_call_count = 0
        repl_errors: list[str] = []
        trajectory: list[dict] = []
        premature_final_rejections = 0
        long_context_mode = len(context) >= self.long_context_threshold

        def llm_query(prompt: str) -> str:
            nonlocal sub_call_count
            sub_call_count += 1
            msgs = [{"role": "user", "content": prompt}]
            response = self.sub_client.complete(msgs)
            if len(response) > _MAX_SUB_RESPONSE_CHARS:
                return response[:_MAX_SUB_RESPONSE_CHARS] + "... [truncated]"
            return response

        self._repl.setup(context, query, llm_query)

        preview = context[:300].replace("\n", " ")
        initial_user = _build_initial_user(len(context), preview, query, long_context_mode)
        history = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": initial_user},
        ]

        for iteration in range(self.max_iterations):
            raw_response = self.root_client.complete(history)
            code = _extract_code(raw_response)
            repeated_strategy = _is_repeated_strategy(code, trajectory)

            stdout, stderr = self._repl.execute(code)
            trajectory.append(
                {
                    "iteration": iteration,
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "final_set": self._repl.is_final,
                }
            )

            if stderr:
                repl_errors.append(stderr[:300])

            if self._repl.is_final:
                rejection_reason = _should_reject_final(
                    context_len=len(context),
                    iteration=iteration + 1,
                    sub_call_count=sub_call_count,
                    rejection_count=premature_final_rejections,
                    long_context_threshold=self.long_context_threshold,
                )
                if rejection_reason:
                    answer_preview = str(self._repl.final_answer)[:200]
                    self._repl.clear_final()
                    premature_final_rejections += 1
                    history.append({"role": "assistant", "content": raw_response})
                    history.append(
                        {
                            "role": "user",
                            "content": _format_premature_final_feedback(
                                rejection_reason=rejection_reason,
                                answer_preview=answer_preview,
                            ),
                        }
                    )
                    continue

                return {
                    "answer": self._repl.final_answer,
                    "iterations": iteration + 1,
                    "sub_calls": sub_call_count,
                    "repl_errors": repl_errors,
                    "timed_out": False,
                    "latency_s": round(time.time() - t0, 2),
                    "trajectory": trajectory,
                }

            consecutive_errors = sum(1 for entry in trajectory[-3:] if entry.get("stderr"))
            history.append({"role": "assistant", "content": raw_response})
            history.append(
                {
                    "role": "user",
                    "content": _format_feedback(
                        stdout=stdout,
                        stderr=stderr,
                        consecutive_errors=consecutive_errors,
                        repeated_strategy=repeated_strategy,
                        long_context_mode=long_context_mode,
                        sub_call_count=sub_call_count,
                    ),
                }
            )

        return {
            "answer": None,
            "iterations": self.max_iterations,
            "sub_calls": sub_call_count,
            "repl_errors": repl_errors,
            "timed_out": True,
            "latency_s": round(time.time() - t0, 2),
            "trajectory": trajectory,
        }


def _build_initial_user(context_len: int, preview: str, query: str, long_context_mode: bool) -> str:
    lines = [
        "Context info:",
        f"  Total length : {context_len:,} characters",
        f"  Preview      : {preview}...",
        "",
        f"Query: {query}",
        "",
        "Write Python code to examine the context and answer the query.",
    ]
    if long_context_mode:
        lines.extend(
            [
                "Long-context mode is active.",
                "Do NOT call FINAL on your first attempt.",
                "First gather evidence with helper functions such as keyword_windows(), regex_windows(), or chunk_text().",
                "Then verify with at least one llm_query() or query_chunks() call before FINAL.",
            ]
        )
    else:
        lines.append("Call FINAL(answer) when you have the answer.")
    return "\n".join(lines)


def _extract_code(response: str) -> str:
    match = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    return response.strip()


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code).strip().lower()


def _is_repeated_strategy(code: str, trajectory: list[dict]) -> bool:
    if not trajectory:
        return False
    current = _normalize_code(code)
    previous = _normalize_code(trajectory[-1].get("code", ""))
    if not current or not previous:
        return False
    similarity = difflib.SequenceMatcher(None, current, previous).ratio()
    return similarity >= _REPEAT_SIMILARITY_THRESHOLD


def _should_reject_final(
    context_len: int,
    iteration: int,
    sub_call_count: int,
    rejection_count: int,
    long_context_threshold: int,
) -> str | None:
    if context_len < long_context_threshold:
        return None
    if rejection_count >= _MAX_PREMATURE_FINAL_REJECTIONS:
        return None
    if iteration == 1:
        return "Long-context answers must not finalize on the first attempt. Gather evidence first."
    if sub_call_count == 0:
        return "Long-context answers must include at least one focused sub-call for verification before FINAL."
    return None


def _format_premature_final_feedback(rejection_reason: str, answer_preview: str) -> str:
    return (
        f"[PREMATURE FINAL REJECTED]\n"
        f"Reason: {rejection_reason}\n"
        f"Your proposed answer was: {answer_preview!r}\n\n"
        f"Next turn requirements:\n"
        f"  1. Gather candidate evidence with keyword_windows(), regex_windows(), or chunk_text()\n"
        f"  2. Print the candidate snippets or extracted evidence\n"
        f"  3. Use at least one llm_query() or query_chunks() call on a focused snippet\n"
        f"  4. Only then call FINAL(answer)\n\n"
        f"Write new Python code now."
    )


def _format_feedback(
    stdout: str,
    stderr: str,
    consecutive_errors: int = 0,
    repeated_strategy: bool = False,
    long_context_mode: bool = False,
    sub_call_count: int = 0,
) -> str:
    parts: list[str] = []

    if stdout:
        truncated = len(stdout) > _MAX_FEEDBACK_CHARS
        preview = stdout[:_MAX_FEEDBACK_CHARS]
        label = f"[stdout | {len(stdout)} chars" + (" | truncated]" if truncated else "]")
        parts.append(f"{label}\n{preview}")

    if stderr:
        error_lines = stderr.strip().splitlines()
        short_error = "\n".join(error_lines[-3:])
        parts.append(
            "[REPL ERROR - your code threw an exception]\n"
            f"{short_error}\n\n"
            "Your previous code did not work. Write a different approach."
        )
        if consecutive_errors >= 2:
            parts.append(
                "[RECOVERY INSTRUCTION]\n"
                "Use a much simpler plan:\n"
                "  1. Inspect head(500) or keyword_windows(...)\n"
                "  2. Print candidate snippets\n"
                "  3. Use llm_query() on one focused snippet\n"
                "  4. FINAL only after evidence is clear"
            )
    elif not stdout:
        parts.append("[no output - code ran silently, no errors]")

    if repeated_strategy:
        parts.append(
            "[REPEATED STRATEGY WARNING]\n"
            "Your latest code is too similar to the previous attempt. Change approach.\n"
            "Use helper functions like chunk_text(), keyword_windows(), regex_windows(), or query_chunks()."
        )

    if long_context_mode and sub_call_count == 0:
        parts.append(
            "[LONG-CONTEXT REMINDER]\n"
            "For long contexts, gather evidence first and verify with at least one llm_query() or query_chunks() call before FINAL."
        )

    parts.append("Write new Python code. Call FINAL(answer) only after the answer is verified.")
    return "\n\n".join(parts)


def _load_system_prompt() -> str:
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "rlm_system.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return _FALLBACK_SYSTEM_PROMPT


_FALLBACK_SYSTEM_PROMPT = """\
You are an RLM (Recursive Language Model). Answer queries about a long document using a Python REPL.

Use helper functions like keyword_windows(), regex_windows(), chunk_text(), llm_query(), and query_chunks().
For long contexts, gather evidence first, then verify with at least one focused sub-call before FINAL.
"""
