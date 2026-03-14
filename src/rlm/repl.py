import io
import json as _json
import re
import sys
import traceback
from typing import Any, Callable


class LocalREPL:
    """
    exec-based Python REPL with persistent namespace.

    The full context P is stored as a variable here — never placed
    in the LLM's context window. The LLM writes Python code cells
    that get executed here iteratively.

    Injects into namespace:
        context     — the full document string
        query       — the question being answered
        context_len — length of context in chars
        llm_query() — recursive sub-LM call function
        FINAL()     — signal final answer as string
        FINAL_VAR() — signal a variable as final answer
        re, json    — pre-imported for convenience
    """

    def __init__(self):
        self._namespace: dict[str, Any] = {}
        self._final_set: bool = False

    def setup(
        self,
        context: str,
        query: str,
        llm_query_fn: Callable[[str], str],
    ) -> None:
        """Initialise namespace. Must be called before execute()."""
        self._final_set = False

        # Capture self reference for closures below
        repl = self

        def FINAL(answer: str) -> None:
            """Signal the final answer. Ends the RLM session."""
            repl._namespace["__final__"] = str(answer)
            repl._final_set = True

        def FINAL_VAR(var: Any) -> None:
            """Signal that a variable holds the final answer."""
            repl._namespace["__final__"] = str(var)
            repl._final_set = True

        self._namespace = {
            "__builtins__": __builtins__,
            # RLM environment
            "context": context,
            "query": query,
            "context_len": len(context),
            "llm_query": llm_query_fn,
            "FINAL": FINAL,
            "FINAL_VAR": FINAL_VAR,
            # Convenience pre-imports
            "re": re,
            "json": _json,
        }

    def execute(self, code: str) -> tuple[str, str]:
        """
        Execute one code cell in the persistent namespace.

        Returns:
            (stdout, stderr) as strings. stderr is non-empty on error.
        """
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        try:
            exec(code, self._namespace)  # noqa: S102
        except SystemExit:
            # Don't let LLM-generated code kill the process
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return "", "SystemExit called — ignored by REPL"
        except Exception:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return "", traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return stdout_buf.getvalue(), stderr_buf.getvalue()

    @property
    def is_final(self) -> bool:
        """True once FINAL() or FINAL_VAR() has been called."""
        return self._final_set

    @property
    def final_answer(self) -> str | None:
        """The answer set by FINAL() / FINAL_VAR(), or None."""
        return self._namespace.get("__final__")

    def user_vars(self) -> list[str]:
        """Variable names created by LLM code (excludes injected names)."""
        _injected = {
            "__builtins__", "__final__",
            "context", "query", "context_len",
            "llm_query", "FINAL", "FINAL_VAR",
            "re", "json",
        }
        return [k for k in self._namespace if k not in _injected]
