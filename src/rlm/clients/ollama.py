import re

import httpx

from .base import BaseLLMClient


class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client using the /api/chat endpoint.

    think=False disables Qwen3's extended reasoning (thinking) mode.
    Thinking is great for hard problems but adds 30–120s overhead per call —
    for our benchmark we compare architectures, so we keep it off.
    Set think=True to re-enable if you want to evaluate reasoning quality.
    """

    def __init__(
        self,
        model: str = "qwen3.5:2b",
        base_url: str = "http://localhost:11434",
        timeout: float = 600.0,      # raised: RLM multi-turn calls can be slow
        temperature: float = 0.0,
        think: bool = False,         # disable Qwen3 thinking mode for speed
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.think = think

    def complete(self, messages: list[dict]) -> str:
        """Call Ollama /api/chat and return the assistant response text."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        # Qwen3-specific: disable extended thinking for faster, consistent inference
        if not self.think:
            payload["think"] = False

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        return _strip_thinking(content)

    def is_available(self) -> bool:
        """Check if the Ollama server is running."""
        try:
            httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            return True
        except Exception:
            return False


def _strip_thinking(text: str) -> str:
    """Strip any residual <think>...</think> blocks (belt-and-suspenders)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
