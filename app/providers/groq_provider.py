"""
Groq concrete implementation of LLMProvider.

Uses the official `groq` SDK for ultra-low-latency inference on open-source models.
Groq's free tier (30 RPM, 14,400 RPD on llama-3.3-70b-versatile) provides a
high-throughput alternative to Gemini, enabling multi-provider resilience.

Features:
  - Async inference via AsyncGroq client
  - Automatic retry with exponential back-off (tenacity)
  - Timeout via SDK configuration
  - JSON-mode via response_format
  - OpenAI-compatible chat completions API
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
    retry_if_exception_type,
)
from groq import AsyncGroq, RateLimitError, APIStatusError, APITimeoutError

from app.providers.llm_provider import LLMProvider, LLMRequest, LLMResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GroqProvider(LLMProvider):
    """
    Production-grade Groq provider.

    Groq delivers sub-200ms inference on LLaMA and Mixtral models via custom
    LPU hardware. The OpenAI-compatible API makes this a drop-in alternative.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> None:
        self._primary_model  = model         or settings.groq_model
        self._fallback_model = fallback_model or settings.groq_fallback_model
        self._active_model   = self._primary_model
        self._client = AsyncGroq(
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
        )
        self._is_degraded: bool = False
        self._degraded_since: float = 0.0

    def model_name(self) -> str:
        return self._active_model

    # ------------------------------------------------------------------
    # Internal call with retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((APITimeoutError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_model(self, model: str, request: LLMRequest) -> LLMResponse:
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
        usage  = response.usage

        prompt_tokens     = usage.prompt_tokens     or 0
        completion_tokens = usage.completion_tokens  or 0
        total_tokens      = usage.total_tokens       or (prompt_tokens + completion_tokens)

        logger.debug(
            "Groq call | model=%s tokens=%d latency=%.0fms",
            model, total_tokens, latency_ms,
        )

        return LLMResponse(
            content=choice.message.content or "",
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            prompt_version=request.prompt_version,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # If we are in degraded mode, skip the primary model to save time
        if self._is_degraded:
            if time.monotonic() - self._degraded_since > 300:  # 5 minute cooldown
                logger.info("Groq degraded cooldown expired. Attempting primary model again.")
                self._is_degraded = False
            else:
                self._active_model = self._fallback_model
                return await self._call_model(self._fallback_model, request)

        try:
            self._active_model = self._primary_model
            return await self._call_model(self._primary_model, request)

        except (RateLimitError, APIStatusError) as exc:
            logger.warning(
                "Primary Groq model %s failed (%s). Entering degraded mode and falling back to %s.",
                self._primary_model, str(exc)[:200], self._fallback_model,
            )
            self._is_degraded = True
            self._degraded_since = time.monotonic()
            self._active_model = self._fallback_model
            return await self._call_model(self._fallback_model, request)
