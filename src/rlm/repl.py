import io
import json as _json
import re
import sys
import traceback
from typing import Any, Callable


class LocalREPL:
    """
    exec-based Python REPL with persistent namespace.

    The full context is stored here as a variable and never placed in the
    model context window. The model writes Python code cells that get
    executed iteratively.
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
        repl = self

        def FINAL(answer: str) -> None:
            repl._namespace["__final__"] = str(answer)
            repl._final_set = True

        def FINAL_VAR(var: Any) -> None:
            repl._namespace["__final__"] = str(var)
            repl._final_set = True

        def head(n: int = 500) -> str:
            return context[: max(0, n)]

        def tail(n: int = 500) -> str:
            if n <= 0:
                return ""
            return context[-n:]

        def context_slice(start: int, end: int) -> str:
            return context[max(0, start): max(0, end)]

        def chunk_text(size: int, overlap: int = 200) -> list[str]:
            if size <= 0:
                raise ValueError("size must be > 0")
            if overlap < 0:
                raise ValueError("overlap must be >= 0")
            step = max(1, size - overlap)
            chunks: list[str] = []
            for start in range(0, len(context), step):
                end = min(len(context), start + size)
                chunks.append(context[start:end])
                if end >= len(context):
                    break
            return chunks

        def keyword_windows(keyword: str, window: int = 400, limit: int = 8) -> list[str]:
            if not keyword:
                return []
            spans: list[str] = []
            seen: set[tuple[int, int]] = set()
            for match in re.finditer(re.escape(keyword), context, re.IGNORECASE):
                start = max(0, match.start() - window)
                end = min(len(context), match.end() + window)
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(context[start:end])
                if len(spans) >= limit:
                    break
            return spans

        def regex_windows(pattern: str, window: int = 400, limit: int = 8, flags: int = 0) -> list[str]:
            if not pattern:
                return []
            spans: list[str] = []
            seen: set[tuple[int, int]] = set()
            for match in re.finditer(pattern, context, flags):
                start = max(0, match.start() - window)
                end = min(len(context), match.end() + window)
                key = (start, end)
                if key in seen:
                    continue
                seen.add(key)
                spans.append(context[start:end])
                if len(spans) >= limit:
                    break
            return spans

        def query_chunks(chunks: list[str], prompt_template: str, limit: int | None = None) -> list[str]:
            selected = chunks if limit is None else chunks[:limit]
            answers: list[str] = []
            for index, chunk in enumerate(selected):
                prompt = prompt_template.format(chunk=chunk, query=query, index=index)
                answers.append(_llm_query_wrapped(prompt))
            return answers

        def _llm_query_wrapped(prompt: str) -> str:
            """Wrapper that auto-prints the sub-LM response."""
            # Append conciseness instruction so sub-LM returns just the answer
            enhanced_prompt = prompt + "\n\nAnswer concisely — return ONLY the answer, no explanation."
            result = llm_query_fn(enhanced_prompt)
            # Auto-print so the model SEES the result in stdout
            print(f"[Sub-LM answered] {result}")
            return result

        self._namespace = {
            "__builtins__": __builtins__,
            "context": context,
            "query": query,
            "context_len": len(context),
            "llm_query": _llm_query_wrapped,
            "query_chunks": query_chunks,
            "FINAL": FINAL,
            "FINAL_VAR": FINAL_VAR,
            "head": head,
            "tail": tail,
            "context_slice": context_slice,
            "chunk_text": chunk_text,
            "keyword_windows": keyword_windows,
            "regex_windows": regex_windows,
            "re": re,
            "json": _json,
        }

    # Names that must never be overwritten by model code
    _PROTECTED_NAMES = {
        "context", "query", "context_len",
        "llm_query", "query_chunks",
        "FINAL", "FINAL_VAR",
        "head", "tail", "context_slice",
        "chunk_text", "keyword_windows", "regex_windows",
    }

    def execute(self, code: str) -> tuple[str, str]:
        """Execute one code cell in the persistent namespace."""
        # Strip function redefinitions that shadow built-in helpers
        code = self._strip_redefinitions(code)

        # Snapshot protected values before exec
        saved = {k: self._namespace[k] for k in self._PROTECTED_NAMES if k in self._namespace}

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        try:
            exec(code, self._namespace)  # noqa: S102
        except SystemExit:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return "", "SystemExit called - ignored by REPL"
        except Exception:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            return "", traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        # Restore any overwritten built-ins
        overwritten = []
        for name, original in saved.items():
            if self._namespace.get(name) is not original:
                self._namespace[name] = original
                overwritten.append(name)

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()

        if overwritten:
            warning = (
                f"[REPL WARNING] You redefined built-in function(s): {', '.join(overwritten)}. "
                f"This was blocked. These functions are ALREADY available — just call them directly. "
                f"Example: results = keyword_windows('search term')\n"
            )
            stderr = warning + stderr

        return stdout, stderr

    def _strip_redefinitions(self, code: str) -> str:
        """Remove 'def func_name(...)' blocks that shadow REPL built-ins."""
        lines = code.split("\n")
        cleaned = []
        skip_indent = None

        for line in lines:
            stripped = line.lstrip()

            # Check if this line starts a function def that shadows a built-in
            if stripped.startswith("def "):
                func_name = stripped[4:].split("(")[0].strip()
                if func_name in self._PROTECTED_NAMES:
                    # Skip this function definition and its body
                    skip_indent = len(line) - len(stripped)
                    continue

            # Skip indented body of a blocked def
            if skip_indent is not None:
                if stripped == "" or (len(line) - len(stripped)) > skip_indent:
                    continue
                else:
                    skip_indent = None

            # Also block direct re-assignment of context
            if stripped.startswith("context =") or stripped.startswith("context="):
                continue

            cleaned.append(line)

        return "\n".join(cleaned)

    def clear_final(self) -> None:
        """Undo a premature FINAL() call so the loop can continue."""
        self._final_set = False
        self._namespace.pop("__final__", None)

    @property
    def is_final(self) -> bool:
        return self._final_set

    @property
    def final_answer(self) -> str | None:
        return self._namespace.get("__final__")

    def user_vars(self) -> list[str]:
        injected = {
            "__builtins__", "__final__",
            "context", "query", "context_len",
            "llm_query", "query_chunks",
            "FINAL", "FINAL_VAR",
            "head", "tail", "context_slice",
            "chunk_text", "keyword_windows", "regex_windows",
            "re", "json",
        }
        return [name for name in self._namespace if name not in injected]
