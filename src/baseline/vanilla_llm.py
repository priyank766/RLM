import time
from typing import Any

from rlm.clients.base import BaseLLMClient

# At ~4 chars/token, 80K chars ≈ 20K tokens — safely within most model windows.
# Adjust down if Ollama runs out of memory on your hardware.
_DEFAULT_MAX_CONTEXT_CHARS = 80_000


class VanillaLLM:
    """
    Standard LLM baseline: full context passed directly into the context window.

    No REPL, no recursion. This is what normal LLM inference looks like.
    Context is truncated with a warning if it exceeds max_context_chars.
    """

    def __init__(
        self,
        client: BaseLLMClient,
        max_context_chars: int = _DEFAULT_MAX_CONTEXT_CHARS,
    ):
        self.client = client
        self.max_context_chars = max_context_chars

    def completion(self, context: str, query: str) -> dict[str, Any]:
        """
        Run vanilla LLM inference.

        Returns a dict with:
            answer            — str
            truncated         — bool (True if context was cut)
            context_chars_used — int
            latency_s         — float
        """
        t0 = time.time()

        truncated = len(context) > self.max_context_chars
        used_context = context[: self.max_context_chars] if truncated else context

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. Answer the query based solely on the "
                    "provided context. Be concise and give a direct answer."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{used_context}\n\nQuery: {query}\n\nAnswer:",
            },
        ]

        answer = self.client.complete(messages)

        return {
            "answer": answer,
            "truncated": truncated,
            "context_chars_used": len(used_context),
            "latency_s": round(time.time() - t0, 2),
        }
