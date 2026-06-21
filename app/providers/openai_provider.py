"""
OpenAI concrete implementation of LLMProvider.

Features:
- Automatic retry with exponential back-off (tenacity)
- Timeout guard
- Fallback model on rate-limit / server errors
- Token counting via tiktoken
- Structured JSON output support
"""
import time
import logging
from typing import Optional

from openai import AsyncOpenAI, RateLimitError, APIStatusError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.providers.llm_provider import LLMProvider, LLMRequest, LLMResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Cost table (USD per 1k tokens) — update as pricing changes
_COST_TABLE = {
    "gpt-4o":             {"input": 0.005,   "output": 0.015},
    "gpt-4o-mini":        {"input": 0.00015, "output": 0.0006},
    "gpt-4-turbo":        {"input": 0.01,    "output": 0.03},
    "gpt-3.5-turbo":      {"input": 0.0005,  "output": 0.0015},
}


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    key = next((k for k in _COST_TABLE if model.startswith(k)), None)
    if not key:
        return 0.0
    rates = _COST_TABLE[key]
    return (prompt_tokens / 1000) * rates["input"] + (completion_tokens / 1000) * rates["output"]


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ):
        self._primary_model = model or settings.openai_model
        self._fallback_model = fallback_model or settings.openai_fallback_model
        self._active_model = self._primary_model
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=settings.llm_timeout_seconds,
        )

    def model_name(self) -> str:
        return self._active_model

    @retry(
        retry=retry_if_exception_type((RateLimitError, APITimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_openai(self, model: str, request: LLMRequest) -> LLMResponse:
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user",   "content": request.user_prompt},
        ]

        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        if request.response_format:
            kwargs["response_format"] = request.response_format

        t0 = time.monotonic()
        response = await self._client.chat.completions.create(**kwargs)
        latency_ms = (time.monotonic() - t0) * 1000

        choice = response.choices[0]
        usage = response.usage

        cost = _calculate_cost(model, usage.prompt_tokens, usage.completion_tokens)

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
            prompt_version=request.prompt_version,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            self._active_model = self._primary_model
            return await self._call_openai(self._primary_model, request)
        except (RateLimitError, APIStatusError) as exc:
            logger.warning(
                "Primary model %s failed (%s). Falling back to %s.",
                self._primary_model, exc, self._fallback_model,
            )
            self._active_model = self._fallback_model
            return await self._call_openai(self._fallback_model, request)
