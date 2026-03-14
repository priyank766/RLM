from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def complete(self, messages: list[dict]) -> str:
        """
        Send a chat completion request and return the response text.

        Args:
            messages: List of {role, content} dicts (OpenAI-style)

        Returns:
            The model's response as a string
        """
        ...
