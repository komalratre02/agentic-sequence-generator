"""
Smart Provider Router — production-grade multi-provider LLM orchestration.

Architecture:
  ┌─────────────┐    ┌───────────────┐    ┌────────────────┐
  │ SmartRouter  │───▶│ GroqProvider  │    │ GeminiProvider │
  │             │    │ (primary)     │    │ (fallback)     │
  │ Circuit     │    └───────────────┘    └────────────────┘
  │ Breaker     │
  │ Health      │
  │ Tracking    │
  └─────────────┘

Features:
  - Health-based provider selection with circuit breaker pattern
  - Automatic cross-provider failover (Groq → Gemini or vice versa)
  - Per-provider latency tracking and success rate monitoring
  - Graceful degradation when a provider is unavailable
  - Provider health exposed via health_report() for observability

Design Decisions:
  - Circuit opens after 3 consecutive failures (configurable)
  - Circuit half-opens after 60s cooldown to probe recovery
  - Provider order is configurable but defaults to Groq (faster) → Gemini
  - All providers share the same LLMProvider interface — zero agent changes
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from app.providers.llm_provider import LLMProvider, LLMRequest, LLMResponse
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Provider Health Tracking
# ---------------------------------------------------------------------------

@dataclass
class ProviderHealth:
    """Real-time health metrics for a single LLM provider."""
    name: str
    consecutive_failures: int = 0
    total_requests: int = 0
    total_failures: int = 0
    total_latency_ms: float = 0.0
    last_failure_time: float = 0.0
    circuit_open: bool = False

    # Configuration
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0

    @property
    def avg_latency_ms(self) -> float:
        successful = self.total_requests - self.total_failures
        return self.total_latency_ms / max(successful, 1)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests

    @property
    def is_available(self) -> bool:
        """Check if provider is available (circuit closed or half-open for probe)."""
        if not self.circuit_open:
            return True
        # Half-open: allow a probe after cooldown
        elapsed = time.monotonic() - self.last_failure_time
        if elapsed >= self.cooldown_seconds:
            return True
        return False

    def record_success(self, latency_ms: float) -> None:
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
        if self.circuit_open:
            logger.info("Circuit CLOSED for %s — recovery confirmed.", self.name)
            self.circuit_open = False

    def record_failure(self) -> None:
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.failure_threshold and not self.circuit_open:
            self.circuit_open = True
            logger.warning(
                "Circuit OPEN for %s — %d consecutive failures.",
                self.name, self.consecutive_failures,
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": "open" if self.circuit_open else ("half-open" if self.circuit_open and self.is_available else "closed"),
            "total_requests": self.total_requests,
            "success_rate": round(self.success_rate, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "consecutive_failures": self.consecutive_failures,
            "is_available": self.is_available,
        }


# ---------------------------------------------------------------------------
# Smart Router
# ---------------------------------------------------------------------------

class SmartRouter(LLMProvider):
    """
    Intelligent multi-provider LLM router with circuit breaker.

    Selects the best available provider based on health status:
      1. Try providers in priority order (fastest first)
      2. Skip providers with open circuits
      3. On failure, mark provider and try next
      4. If all fail, raise the last exception

    Usage:
        router = SmartRouter()
        response = await router.complete(request)
        print(router.health_report())  # Per-provider metrics
    """

    def __init__(self) -> None:
        self._providers: list[tuple[LLMProvider, ProviderHealth]] = []
        self._active_model: str = "initializing"
        self._active_provider_name: str = "none"
        self._initialize_providers()

    def _initialize_providers(self) -> None:
        """Dynamically initialize available providers based on config."""
        # Groq — preferred for speed (sub-200ms inference)
        if settings.groq_api_key:
            from app.providers.groq_provider import GroqProvider
            provider = GroqProvider()
            health = ProviderHealth(name="groq")
            self._providers.append((provider, health))
            logger.info("SmartRouter registered: Groq (%s)", settings.groq_model)

        # Gemini — reliable fallback with generous free tier
        if settings.google_api_key:
            from app.providers.gemini_provider import GeminiProvider
            provider = GeminiProvider()
            health = ProviderHealth(name="gemini")
            self._providers.append((provider, health))
            logger.info("SmartRouter registered: Gemini (%s)", settings.gemini_model)

        if not self._providers:
            raise RuntimeError(
                "No LLM providers configured. Set GROQ_API_KEY and/or GOOGLE_API_KEY."
            )

        self._active_model = self._providers[0][0].model_name()
        self._active_provider_name = self._providers[0][1].name

    def model_name(self) -> str:
        return self._active_model

    @property
    def active_provider_name(self) -> str:
        return self._active_provider_name

    def health_report(self) -> list[dict]:
        """Return health metrics for all registered providers."""
        return [h.to_dict() for _, h in self._providers]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """
        Route to the best available provider with automatic failover.

        Tries each provider in priority order, skipping those with open
        circuits. On success, records latency; on failure, records the
        failure and tries the next provider.
        """
        last_exception: Optional[Exception] = None

        for provider, health in self._providers:
            if not health.is_available:
                logger.debug(
                    "Skipping %s — circuit open (cooldown %.0fs remaining).",
                    health.name,
                    max(0, health.cooldown_seconds - (time.monotonic() - health.last_failure_time)),
                )
                continue

            try:
                response = await provider.complete(request)
                health.record_success(response.latency_ms)
                self._active_model = provider.model_name()
                self._active_provider_name = health.name
                return response

            except Exception as exc:
                health.record_failure()
                last_exception = exc
                logger.warning(
                    "Provider %s failed: %s — trying next provider.",
                    health.name, str(exc)[:200],
                )

        # All providers exhausted
        logger.error("All LLM providers failed. Last error: %s", last_exception)
        raise last_exception or RuntimeError("No LLM providers available.")
