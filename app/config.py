"""Configuration utilities for the QA service."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional
import os

import yaml
from pydantic import BaseModel, Field


class OpenAISettings(BaseModel):
    """Settings required to talk to the OpenAI API."""

    api_key: str = Field(..., description="Secret key for OpenAI API access")
    model: str = Field(
        default="gpt-5-mini-2025-08-07",
        description="Model identifier to use for answering questions.",
    )
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    max_output_tokens: int = Field(300, ge=64, le=2048)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model identifier used for embedding member messages.",
    )
    embedding_batch_size: int = Field(
        100, ge=1, le=2048, description="Maximum messages to embed per request."
    )
    reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
    verbosity: Literal["low", "medium", "high"] = "medium"
    trace_output: bool = Field(
        False, description="Print raw model output whenever the LLM responds."
    )


class MessagesAPISettings(BaseModel):
    """Settings for the upstream messages API."""

    base_url: str = Field(..., description="Base URL for the member messages API")
    skip: int = Field(0, ge=0, description="Number of messages to skip from the top")
    limit: int = Field(200, ge=1, le=1000, description="Total messages to retrieve")
    timeout_seconds: float = Field(10.0, gt=0)


class RetrievalSettings(BaseModel):
    """Settings that govern how many messages are retrieved for the LLM."""

    top_k: int = Field(8, ge=1, le=100, description="Messages to feed into the LLM")


class Settings(BaseModel):
    """Top-level application settings loaded from a YAML config file."""

    app_name: str = "QA Service"
    openai: OpenAISettings
    messages_api: MessagesAPISettings
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings) # type: ignore


_SETTINGS_CACHE: Optional[Settings] = None


def _read_raw_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must define a mapping at the top level")

    return data


def load_settings(path: str | os.PathLike[str] | None = None, *, force_reload: bool = False) -> Settings:
    """Load settings from disk, with optional caching."""

    global _SETTINGS_CACHE

    if _SETTINGS_CACHE is not None and not force_reload and path is None:
        return _SETTINGS_CACHE

    resolved_path = Path(
        path
        or os.environ.get("QA_SERVICE_CONFIG")
        or Path("config/settings.yaml")
    ).expanduser().resolve()

    raw_data = _read_raw_config(resolved_path)
    settings = Settings(**raw_data)

    if path is None and not force_reload:
        _SETTINGS_CACHE = settings

    return settings


__all__ = [
    "Settings",
    "OpenAISettings",
    "MessagesAPISettings",
    "RetrievalSettings",
    "load_settings",
]
