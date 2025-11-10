"""Service layer that coordinates the pipeline."""
from __future__ import annotations

import asyncio
from typing import List, Tuple

from .embeddings import EmbeddingsClient, VectorizedMessage, cosine_similarity
from .llm import LLMClient
from .message_client import MessageRecord, MessagesClient


class QAService:
    def __init__(
        self,
        messages_client: MessagesClient,
        embeddings_client: EmbeddingsClient,
        llm_client: LLMClient,
        retrieval_top_k: int,
        *,
        message_cache_limit: int | None = None,
    ):
        self._messages_client = messages_client
        self._embeddings_client = embeddings_client
        self._llm_client = llm_client
        self._top_k = retrieval_top_k
        self._cache_limit = message_cache_limit
        self._vectorized_messages: List[VectorizedMessage] = []
        self._cached_messages: List[MessageRecord] = []
        self._cache_lock = asyncio.Lock()
        self._cache_ready = asyncio.Event()

    async def answer_question(
        self, question: str, *, reasoning_effort: str | None = None
    ) -> Tuple[str, int]:
        await self._ensure_vectors_ready()
        if not self._vectorized_messages:
            return "No member messages are available at the moment.", 0

        question_vector = await self._embeddings_client.embed_question(question)
        top_messages = self._select_top_messages(
            question_vector, self._vectorized_messages
        )
        answer = await self._llm_client.answer(
            question,
            [vector.record for vector in top_messages],
            reasoning_effort=reasoning_effort,
        )
        return answer, len(top_messages)

    async def warm_cache(self, *, force: bool = False) -> None:
        if self._cache_ready.is_set() and not force:
            return

        async with self._cache_lock:
            if self._cache_ready.is_set() and not force:
                return

            self._cache_ready.clear()
            messages = await self._messages_client.fetch_messages(
                limit=self._cache_limit
            )
            if not messages:
                self._vectorized_messages = []
                self._cached_messages = []
                self._cache_ready.set()
                return

            vectorized = await self._embeddings_client.embed_messages(messages)
            self._vectorized_messages = vectorized
            self._cached_messages = list(messages)
            self._cache_ready.set()

    async def _ensure_vectors_ready(self) -> None:
        if self._cache_ready.is_set():
            return
        await self.warm_cache()

    async def get_cached_messages(self) -> List[MessageRecord]:
        await self._ensure_vectors_ready()
        return list(self._cached_messages)

    def _select_top_messages(
        self, question_vector: List[float], vectorized_messages: List[VectorizedMessage]
    ) -> List[VectorizedMessage]:
        if not vectorized_messages:
            return []

        scored = [
            (cosine_similarity(question_vector, item.vector), item)
            for item in vectorized_messages
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        limit = min(self._top_k, len(scored))
        return [item for _, item in scored[:limit]]


__all__ = ["QAService"]
