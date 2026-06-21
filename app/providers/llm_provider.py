"""
Abstract LLM Provider — all agents depend on this interface, never on a concrete SDK.

Adding a new provider (Anthropic, OpenAI, etc.) only requires implementing these two methods.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    prompt_version: Optional[str] = None


@dataclass
class LLMRequest:
    system_prompt: str
    user_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2048
    prompt_version: Optional[str] = None
    response_format: Optional[dict] = None   # {"type": "json_object"} for JSON mode


class LLMProvider(ABC):
    """Contract every LLM backend must fulfil."""

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a chat completion and return a normalised response."""
        ...

    @abstractmethod
    def model_name(self) -> str:
        """Return the currently active model identifier."""
        ...
