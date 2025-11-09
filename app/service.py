"""Service layer that coordinates the pipeline."""
from __future__ import annotations

import asyncio
from typing import List, Tuple

from .embeddings import EmbeddingsClient, VectorizedMessage, cosine_similarity
from .llm import LLMClient
from .message_client import MessagesClient


class QAService:
    def __init__(
        self,
        messages_client: MessagesClient,
        embeddings_client: EmbeddingsClient,
        llm_client: LLMClient,
        retrieval_top_k: int,
    ):
        self._messages_client = messages_client
        self._embeddings_client = embeddings_client
        self._llm_client = llm_client
        self._top_k = retrieval_top_k

    async def answer_question(self, question: str) -> Tuple[str, int]:
        messages = await self._messages_client.fetch_messages()
        if not messages:
            return "No member messages are available at the moment.", 0

        vector_task = self._embeddings_client.embed_messages(messages)
        question_task = self._embeddings_client.embed_question(question)
        vectorized, question_vector = await asyncio.gather(vector_task, question_task)

        top_messages = self._select_top_messages(question_vector, vectorized)
        answer = await self._llm_client.answer(
            question, [vector.record for vector in top_messages]
        )
        return answer, len(top_messages)

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
