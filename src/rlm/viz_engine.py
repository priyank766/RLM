"""
RLM Visualization Server — see the RLM process step-by-step in your browser.

Runs a FastAPI server with WebSocket for real-time streaming of RLM events.
The frontend shows each iteration: root LM output, code, REPL result, sub-calls.

Usage:
    uv run python scripts/run_viz.py
    uv run python scripts/run_viz.py --port 8765
    uv run python scripts/run_viz.py --model qwen3.5:2b
"""

import asyncio
import json
import time
import re
import difflib
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rlm.clients.base import BaseLLMClient
from rlm.repl import LocalREPL


# ---------------------------------------------------------------------------
# Observable RLM_REPL — emits events at every step
# ---------------------------------------------------------------------------

_MAX_FEEDBACK_CHARS = 600
_MAX_SUB_RESPONSE_CHARS = 1200
_LONG_CONTEXT_THRESHOLD = 16_000
_MAX_PREMATURE_FINAL_REJECTIONS = 2
_REPEAT_SIMILARITY_THRESHOLD = 0.92


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


def _load_system_prompt() -> str:
    # viz_engine.py is at src/rlm/ — go up to project root (3 levels)
    project_root = Path(__file__).parent.parent.parent
    prompt_path = project_root / "prompts" / "rlm_system.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8").strip()
    return (
        "You are an RLM (Recursive Language Model). Answer queries about a long "
        "document using a Python REPL.\n\nUse helper functions like keyword_windows(), "
        "regex_windows(), chunk_text(), llm_query(), and query_chunks().\nFor long "
        "contexts, gather evidence first, then verify with at least one focused "
        "sub-call before FINAL."
    )


async def run_rlm_observable(
    root_client: BaseLLMClient,
    sub_client: BaseLLMClient,
    context: str,
    query: str,
    emit: Callable,  # async callable that sends events to WebSocket
    max_iterations: int = 20,
):
    """
    Run the RLM loop, emitting events at every step for the frontend.

    Events emitted:
        init          — run started, context info
        root_thinking — root LM is generating (before response)
        root_response — root LM responded with code
        repl_exec     — REPL executed the code
        sub_call      — a sub-LM call happened inside REPL
        final_reject  — premature FINAL was rejected
        final         — RLM finished with answer
        timeout       — max iterations reached
        error         — something went wrong
    """
    t0 = time.time()
    sub_call_count = 0
    repl_errors = []
    trajectory = []
    premature_final_rejections = 0
    long_context_mode = len(context) >= _LONG_CONTEXT_THRESHOLD
    system_prompt = _load_system_prompt()
    repl = LocalREPL()

    # Sub-call events list for the current iteration
    current_sub_calls = []

    def llm_query(prompt: str) -> str:
        nonlocal sub_call_count
        sub_call_count += 1
        call_idx = sub_call_count

        # We can't await inside a sync callback, so we run in the event loop
        # This is called from exec() which is sync, so we use a thread-safe approach
        msgs = [{"role": "user", "content": prompt}]
        response = sub_client.complete(msgs)
        if len(response) > _MAX_SUB_RESPONSE_CHARS:
            response = response[:_MAX_SUB_RESPONSE_CHARS] + "... [truncated]"

        current_sub_calls.append({
            "call_index": call_idx,
            "prompt": prompt[:500],
            "response": response[:800],
        })
        return response

    repl.setup(context, query, llm_query)

    # Emit init
    preview = context[:500].replace("\n", " ")
    await emit({
        "type": "init",
        "context_length": len(context),
        "context_preview": preview,
        "query": query,
        "long_context_mode": long_context_mode,
        "max_iterations": max_iterations,
    })

    # Build initial history
    initial_lines = [
        f"Context loaded: {len(context):,} characters.",
        f"Preview: {preview[:200]}...",
        "",
        f"Query: {query}",
        "",
        "IMPORTANT: All helper functions are ALREADY loaded. Just call them directly:",
        "  keyword_windows('search_term')  → finds text around matches",
        "  llm_query('your question...')    → asks sub-LM, returns answer string",
        "  FINAL('your answer')             → submits final answer",
        "",
        "DO NOT redefine these functions. DO NOT create your own context variable.",
        "Write short Python code (under 10 lines). Example:",
        "  results = keyword_windows('password', window=500)",
        "  print(results[0][:300])",
    ]
    if long_context_mode:
        initial_lines.extend([
            "",
            "LONG CONTEXT MODE: Do NOT call FINAL yet.",
            "Step 1: Search with keyword_windows() or regex_windows()",
            "Step 2: Verify with llm_query() on a snippet",
            "Step 3: Only then call FINAL(answer)",
        ])
    else:
        initial_lines.append("\nCall FINAL(answer) when you have the answer.")

    initial_user = "\n".join(initial_lines)
    history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_user},
    ]

    for iteration in range(max_iterations):
        current_sub_calls = []

        # Root LM thinking
        await emit({
            "type": "root_thinking",
            "iteration": iteration + 1,
        })

        # Get root LM response
        try:
            raw_response = root_client.complete(history)
        except Exception as e:
            await emit({
                "type": "error",
                "iteration": iteration + 1,
                "message": f"Root LM call failed: {e}",
            })
            return

        code = _extract_code(raw_response)

        await emit({
            "type": "root_response",
            "iteration": iteration + 1,
            "raw_response": raw_response[:2000],
            "code": code,
        })

        # Check for repeated strategy
        repeated = False
        if trajectory:
            current_norm = _normalize_code(code)
            prev_norm = _normalize_code(trajectory[-1].get("code", ""))
            if current_norm and prev_norm:
                sim = difflib.SequenceMatcher(None, current_norm, prev_norm).ratio()
                repeated = sim >= _REPEAT_SIMILARITY_THRESHOLD

        # Execute in REPL
        stdout, stderr = repl.execute(code)

        trajectory.append({
            "iteration": iteration,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "final_set": repl.is_final,
        })

        if stderr:
            repl_errors.append(stderr[:300])

        # Emit REPL result + any sub-calls
        await emit({
            "type": "repl_exec",
            "iteration": iteration + 1,
            "stdout": stdout[:1500] if stdout else "",
            "stderr": stderr[:1000] if stderr else "",
            "sub_calls": current_sub_calls,
            "sub_call_total": sub_call_count,
            "final_set": repl.is_final,
        })

        # Handle FINAL
        if repl.is_final:
            # Check for premature final rejection
            rejection_reason = None
            if len(context) >= _LONG_CONTEXT_THRESHOLD:
                if premature_final_rejections < _MAX_PREMATURE_FINAL_REJECTIONS:
                    if iteration == 0:
                        rejection_reason = "Long-context answers must not finalize on the first attempt."
                    elif sub_call_count == 0:
                        rejection_reason = "Must include at least one sub-call before FINAL."

            if rejection_reason:
                answer_preview = str(repl.final_answer)[:200]
                repl.clear_final()
                premature_final_rejections += 1

                await emit({
                    "type": "final_reject",
                    "iteration": iteration + 1,
                    "reason": rejection_reason,
                    "answer_preview": answer_preview,
                })

                # Include sub-call evidence in rejection feedback
                sub_evidence = ""
                if current_sub_calls:
                    last_response = current_sub_calls[-1].get("response", "")
                    sub_evidence = f"\nYour sub-LM found: {last_response[:300]}\nUse the sub-LM's finding as your answer."

                history.append({"role": "assistant", "content": raw_response})
                history.append({
                    "role": "user",
                    "content": (
                        f"[PREMATURE FINAL REJECTED]\n"
                        f"Reason: {rejection_reason}\n"
                        f"Your proposed answer was: {answer_preview!r}\n"
                        f"{sub_evidence}\n\n"
                        f"Write new code. Use the sub-LM evidence to call FINAL with the CORRECT answer."
                    ),
                })
                continue

            # Accepted FINAL
            latency = round(time.time() - t0, 2)
            await emit({
                "type": "final",
                "iteration": iteration + 1,
                "answer": str(repl.final_answer),
                "total_iterations": iteration + 1,
                "total_sub_calls": sub_call_count,
                "repl_errors": len(repl_errors),
                "latency_s": latency,
            })
            return

        # Build feedback for next iteration
        parts = []
        if stdout:
            truncated = len(stdout) > _MAX_FEEDBACK_CHARS
            label = f"[stdout | {len(stdout)} chars" + (" | truncated]" if truncated else "]")
            parts.append(f"{label}\n{stdout[:_MAX_FEEDBACK_CHARS]}")

        if stderr:
            error_lines = stderr.strip().splitlines()
            short_error = "\n".join(error_lines[-3:])
            parts.append(f"[REPL ERROR]\n{short_error}\nWrite a different approach.")
        elif not stdout:
            parts.append("[no output - code ran silently]")

        if repeated:
            parts.append("[REPEATED STRATEGY WARNING] Change your approach.")

        if long_context_mode and sub_call_count == 0:
            parts.append("[LONG-CONTEXT REMINDER] Use llm_query() before FINAL.")

        parts.append("Write new Python code. Call FINAL(answer) when verified.")
        feedback = "\n\n".join(parts)

        history.append({"role": "assistant", "content": raw_response})
        history.append({"role": "user", "content": feedback})

    # Timeout
    latency = round(time.time() - t0, 2)
    await emit({
        "type": "timeout",
        "total_iterations": max_iterations,
        "total_sub_calls": sub_call_count,
        "repl_errors": len(repl_errors),
        "latency_s": latency,
    })
