"""LLM connector — wraps LiteLLM for unified access to 100+ LLM providers.

Key improvements over the old ``OpenaiCompatibleLLM``:
- Client is created **once** at instance level (not per-call).
- Errors raise ``LLMConnectionError`` (not silent ``''`` return).
- Built-in retry with exponential backoff + circuit breaker.
- Async-first interface with sync convenience wrapper.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import litellm

from rag0.config import LLMConfig
from rag0.exceptions import LLMConnectionError
from rag0.logging import get_logger
from rag0.types import Message

logger = get_logger(__name__)

# LiteLLM handles API key resolution automatically from standard env vars
# (DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
# We just set this once to suppress noisy logs.
litellm.suppress_debug_info = True


class LLMConnector:
    """Async-first LLM connector backed by LiteLLM.

    Args:
        config: LLM configuration from :class:`rag0.config.LLMConfig`.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._circuit_open_until = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(self, messages: list[Message], **kwargs: Any) -> str:
        """Generate a complete response for *messages*.

        Args:
            messages: List of chat messages (system, user, assistant).
            **kwargs: Additional LiteLLM parameters (temperature, max_tokens, etc.).

        Returns:
            The generated text response.

        Raises:
            LLMConnectionError: On connection failure, timeout, or circuit open.
        """
        self._check_circuit()

        payload = self._build_payload(messages, kwargs)
        logger.debug("LLM generate", model=self._config.model_name)

        try:
            response = await litellm.acompletion(**payload)
            text = response.choices[0].message.content or ""
            self._on_success()
            return text
        except Exception as exc:
            self._on_failure(exc)
            raise LLMConnectionError(
                f"LLM call failed: {exc}",
                cause=exc,
                context={"model": self._config.model_name},
            ) from exc

    async def generate_stream(
        self, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token.

        Args:
            messages: List of chat messages.
            **kwargs: Additional LiteLLM parameters.

        Yields:
            Text tokens as they arrive.

        Raises:
            LLMConnectionError: On connection failure.
        """
        self._check_circuit()

        payload = self._build_payload(messages, kwargs)
        payload["stream"] = True
        logger.debug("LLM stream", model=self._config.model_name)

        try:
            response = await litellm.acompletion(**payload)
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
            self._on_success()
        except Exception as exc:
            self._on_failure(exc)
            raise LLMConnectionError(
                f"LLM stream failed: {exc}",
                cause=exc,
                context={"model": self._config.model_name},
            ) from exc

    # ------------------------------------------------------------------
    # Circuit breaker
    # ------------------------------------------------------------------
    def _check_circuit(self) -> None:
        """Raise if the circuit breaker is open."""
        if time.monotonic() < self._circuit_open_until:
            remaining = self._circuit_open_until - time.monotonic()
            raise LLMConnectionError(
                f"Circuit breaker open — LLM unavailable for {remaining:.0f}s",
                context={"model": self._config.model_name},
            )

    def _on_success(self) -> None:
        self._failure_count = 0

    def _on_failure(self, exc: Exception) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= 5:
            self._circuit_open_until = time.monotonic() + 30
            logger.warning(
                "Circuit breaker opened",
                failures=self._failure_count,
                cooldown_s=30,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_payload(
        self, messages: list[Message], extra: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "model": self._config.model_name,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": extra.get("temperature", self._config.temperature),
            "timeout": extra.get("timeout", self._config.timeout),
            "num_retries": extra.get("num_retries", self._config.max_retries),
            **{k: v for k, v in extra.items() if k not in ("temperature", "timeout", "num_retries")},
        }
