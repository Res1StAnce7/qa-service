"""Pydantic schemas shared by the FastAPI layer."""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel, Field


class AnswerResponse(BaseModel):
    """Outbound payload returned to clients."""

    answer: str
    sources_used: int = Field(..., description="Number of upstream messages provided to the LLM")


class MessageSchema(BaseModel):
    """Public message payload returned by /messages."""

    user_name: str
    timestamp: datetime
    message: str
