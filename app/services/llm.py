from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class LLMClient:
    """Wrapper around AsyncOpenAI configured for OpenRouter."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key or settings.llm_api_key
        self._base_url = base_url or OPENROUTER_BASE_URL
        self._model = model or settings.llm_model
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialised AsyncOpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "LLM API key is not configured. Set CODETUTOR_LLM_API_KEY in your environment or .env file."
                )
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=0,
            )
            logger.info(
                "LLM client initialised (base_url=%s, model=%s, timeout=%ss)",
                self._base_url,
                self._model,
                self._timeout,
            )
        return self._client

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **extra: Any,
    ) -> str:
        """Send a chat completion request and return the assistant message content.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max_tokens.
            **extra: Additional keyword arguments passed to the API.

        Returns:
            The text content of the assistant's response.

        Raises:
            ValueError: If the API key is not configured.
            openai.APIError: On upstream API errors.
        """
        logger.info("LLM request starting (model=%s, messages=%d)", model or self._model, len(messages))
        response = await self.client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature if temperature is not None else self._temperature,
            max_tokens=max_tokens or self._max_tokens,
            **extra,
        )
        content = response.choices[0].message.content
        logger.info("LLM response received (model=%s, usage=%s)", response.model, response.usage)
        return content

    async def close(self) -> None:
        """Close the underlying client connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("LLM client closed")

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


# Module-level singleton for convenience
llm_client = LLMClient()
