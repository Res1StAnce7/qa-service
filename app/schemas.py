"""Pydantic schemas shared by the FastAPI layer."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Outbound payload returned to clients."""

    answer: str
    sources_used: int = Field(..., description="Number of upstream messages provided to the LLM")


class AskRequest(BaseModel):
    """Inbound payload for POST /ask."""

    question: str = Field(
        ...,
        min_length=1,
        description="Natural language question to answer.",
    )
    reasoning_effort: str | None = Field(
        default=None,
        description="Optional reasoning setting (minimal|low|medium|high).",
    )


class MessageSchema(BaseModel):
    """Public message payload returned by /messages."""

    user_name: str
    timestamp: datetime
    message: str
