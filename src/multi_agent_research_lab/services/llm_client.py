"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client implementation supporting Gemini and OpenAI."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._gemini_client: object | None = None
        self._openai_client: object | None = None

        if self.settings.gemini_api_key:
            from google import genai

            self._gemini_client = genai.Client(api_key=self.settings.gemini_api_key)
        elif self.settings.openai_api_key:
            import openai

            self._openai_client = openai.OpenAI(api_key=self.settings.openai_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with token usage and cost tracking."""
        if self._gemini_client is not None:
            return self._complete_gemini(system_prompt, user_prompt)
        elif self._openai_client is not None:
            return self._complete_openai(system_prompt, user_prompt)
        else:
            return self._complete_mock(system_prompt, user_prompt)

    def _complete_gemini(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        from google import genai
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_prompt if system_prompt else None,
            temperature=0.3,
        )

        client: genai.Client = self._gemini_client  # type: ignore[assignment]
        model = self.settings.gemini_model or "gemini-3.1-flash-lite"

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=config,
        )

        content = response.text or ""
        in_tokens = 0
        out_tokens = 0

        if response.usage_metadata:
            in_tokens = response.usage_metadata.prompt_token_count or 0
            out_tokens = response.usage_metadata.candidates_token_count or 0

        # Cost estimation: ~$0.075 / 1M in, ~$0.30 / 1M out for Gemini flash-lite
        cost = (in_tokens * 0.075 + out_tokens * 0.30) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
        )

    def _complete_openai(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        import openai

        client: openai.OpenAI = self._openai_client  # type: ignore[assignment]
        model = self.settings.openai_model or "gpt-4o-mini"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,  # type: ignore[arg-type]
            temperature=0.3,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        in_tokens = response.usage.prompt_tokens if response.usage else 0
        out_tokens = response.usage.completion_tokens if response.usage else 0
        # Cost estimation: ~$0.15 / 1M in, ~$0.60 / 1M out for gpt-4o-mini
        cost = (in_tokens * 0.15 + out_tokens * 0.60) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
        )

    def _complete_mock(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        prompt_len = len(system_prompt) + len(user_prompt)
        in_tokens = max(1, prompt_len // 4)
        out_tokens = 50
        cost = 0.0
        return LLMResponse(
            content=f"Mock response generated for query: {user_prompt[:50]}...",
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            cost_usd=cost,
        )
