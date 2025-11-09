"""OpenAI helper that turns member messages into answers."""
from __future__ import annotations

import asyncio
from typing import Sequence

from openai import OpenAI
from openai import APIConnectionError, APIStatusError

from .config import OpenAISettings
from .message_client import MessageRecord


def _format_context(messages: Sequence[MessageRecord]) -> str:
    lines = []
    for record in messages:
        lines.append(
            f"- {record.timestamp.isoformat()} | {record.user_name}: {record.message}"
        )
    return "\n".join(lines) if lines else "(no messages provided)"


class LLMClient:
    """Wrapper around OpenAI's chat completions API."""

    def __init__(self, settings: OpenAISettings):
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key)
        self._system_prompt = (
            "You are a meticulous concierge analyst. "
            "Answer questions strictly using the provided member messages. "
            "If the answer cannot be found, reply that the information is unavailable."
        )

    def _invoke(self, question: str, context: str) -> str:
        completion = self._client.responses.create(
            model=self._settings.model,
            reasoning={"effort": self._settings.reasoning_effort},
            text={"verbosity": self._settings.verbosity},
            input=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Messages:\n" + context + "\n\n" + f"Question: {question}\nAnswer:"
                    ),
                },
            ],
        )
        choice = completion.output_text
        return choice if choice else "No answer returned."

    async def answer(self, question: str, messages: Sequence[MessageRecord]) -> str:
        context = _format_context(messages)
        try:
            return await asyncio.to_thread(self._invoke, question, context)
        except (APIConnectionError, APIStatusError) as exc:
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc


__all__ = ["LLMClient"]
