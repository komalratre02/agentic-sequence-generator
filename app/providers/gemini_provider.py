"""
Gemini concrete implementation of LLMProvider.

Uses the official `google-genai` SDK (not the deprecated google-generativeai).
langchain-google-genai is still used for embeddings; all chat completions
go through this class so agents stay SDK-agnostic.

Features:
  - Async inference via google.genai AsyncClient
  - Automatic retry with exponential back-off (tenacity)
  - Timeout via asyncio.wait_for
  - Fallback to gemini-2.5-flash-lite on quota / server errors
  - JSON-mode via response_mime_type="application/json"
  - Precise token counts from usage_metadata
"""
from __future__ import annotations
from google.genai.errors import ClientError

import asyncio
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
from google import genai
from google.genai import types as genai_types

from app.providers.llm_provider import LLMProvider, LLMRequest, LLMResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiProvider(LLMProvider):
    """
    Production-grade Gemini provider.

    Swap to OpenAIProvider or AnthropicProvider by changing which class is
    instantiated in routes.py — zero changes to agent code.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> None:
        self._primary_model  = model         or settings.gemini_model
        self._fallback_model = fallback_model or settings.gemini_fallback_model
        self._active_model   = self._primary_model
        self._client = genai.Client(api_key=settings.google_api_key)
        self._is_degraded: bool = False
        self._degraded_since: float = 0.0

    def model_name(self) -> str:
        return self._active_model

    # ------------------------------------------------------------------
    # Internal call with retry
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type((asyncio.TimeoutError,)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_model(self, model: str, request: LLMRequest) -> LLMResponse:
        # Build system instruction
        system_instruction = request.system_prompt
        if request.response_format and request.response_format.get("type") == "json_object":
            system_instruction += (
                "\n\nIMPORTANT: Your response MUST be valid JSON only. "
                "No markdown code fences, no explanation, no trailing text."
            )

        # Configure generation
        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=request.temperature,
            max_output_tokens=request.max_tokens,
            response_mime_type=(
                "application/json"
                if request.response_format and request.response_format.get("type") == "json_object"
                else "text/plain"
            ),
        )

        t0 = time.monotonic()
        response = await asyncio.wait_for(
            self._client.aio.models.generate_content(
                model=model,
                contents=request.user_prompt,
                config=config,
            ),
            timeout=settings.llm_timeout_seconds,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        content = response.text or ""

        # Extract token usage
        usage = response.usage_metadata
        prompt_tokens     = getattr(usage, "prompt_token_count",     0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens      = getattr(usage, "total_token_count",      0) or (prompt_tokens + completion_tokens)

        logger.debug(
            "Gemini call | model=%s tokens=%d latency=%.0fms",
            model, total_tokens, latency_ms,
        )

        return LLMResponse(
            content=content,
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
                logger.info("Gemini degraded cooldown expired. Attempting primary model again.")
                self._is_degraded = False
            else:
                self._active_model = self._fallback_model
                return await self._call_model(self._fallback_model, request)

        try:
            self._active_model = self._primary_model
            return await self._call_model(self._primary_model, request)

        except ClientError as exc:
            logger.warning(
                "Primary Gemini model %s failed (%s). Entering degraded mode and falling back to %s.",
                self._primary_model, str(exc)[:200], self._fallback_model,
            )
            self._is_degraded = True
            self._degraded_since = time.monotonic()
            self._active_model = self._fallback_model
            return await self._call_model(self._fallback_model, request)
