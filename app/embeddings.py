"""Utilities for turning text into embedding vectors."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI

from .config import OpenAISettings
from .message_client import MessageRecord


@dataclass(slots=True)
class VectorizedMessage:
    record: MessageRecord
    vector: List[float]


class EmbeddingsClient:
    """Lightweight batching helper around OpenAI's embeddings API."""

    def __init__(self, settings: OpenAISettings):
        self._settings = settings
        self._client = OpenAI(api_key=settings.api_key)

    async def embed_messages(self, messages: Sequence[MessageRecord]) -> List[VectorizedMessage]:
        if not messages:
            return []
        texts = [self._format_message(record) for record in messages]
        vectors = await self._embed_texts(texts)
        return [VectorizedMessage(record=record, vector=vector) for record, vector in zip(messages, vectors)]

    async def embed_question(self, question: str) -> List[float]:
        vectors = await self._embed_texts([question])
        return vectors[0]

    async def _embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for chunk in _chunk(texts, self._settings.embedding_batch_size):
            batch = await asyncio.to_thread(self._create_embeddings, list(chunk))
            vectors.extend(batch)
        return vectors

    def _create_embeddings(self, batch: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self._settings.embedding_model,
            input=batch,
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _format_message(record: MessageRecord) -> str:
        return (
            f"Timestamp: {record.timestamp.isoformat()}\n"
            f"Member: {record.user_name}\n"
            f"Message: {record.message}"
        )
def _chunk(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be the same length for cosine similarity")

    numerator = 0.0
    sum_a = 0.0
    sum_b = 0.0
    for a, b in zip(vector_a, vector_b):
        numerator += a * b
        sum_a += a * a
        sum_b += b * b

    denominator = math.sqrt(sum_a) * math.sqrt(sum_b)
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = ["EmbeddingsClient", "VectorizedMessage", "cosine_similarity"]
