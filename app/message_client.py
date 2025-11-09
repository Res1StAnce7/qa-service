"""HTTP client for retrieving member messages."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

import httpx

from .config import MessagesAPISettings


@dataclass(slots=True)
class MessageRecord:
    id: str
    user_id: str
    user_name: str
    timestamp: datetime
    message: str

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "MessageRecord":
        ts = payload.get("timestamp")
        if not isinstance(ts, str):
            raise ValueError("Message timestamp missing or invalid")

        iso = ts.replace("Z", "+00:00")
        timestamp = datetime.fromisoformat(iso)

        return cls(
            id=str(payload.get("id")),
            user_id=str(payload.get("user_id")),
            user_name=str(payload.get("user_name")),
            timestamp=timestamp,
            message=str(payload.get("message", "")),
        )


class MessagesClient:
    """Thin wrapper around the upstream messages API."""

    def __init__(self, settings: MessagesAPISettings):
        self._settings = settings

    async def fetch_messages(self, *, limit: int | None = None) -> List[MessageRecord]:
        """Fetch recent messages according to settings."""

        records: List[MessageRecord] = []
        timeout = httpx.Timeout(self._settings.timeout_seconds)

        effective_limit = limit or self._settings.limit
        params = {"skip": self._settings.skip, "limit": effective_limit}
        headers = {"Accept": "application/json"}

        async with httpx.AsyncClient(
            base_url=self._settings.base_url,
            timeout=timeout,
            follow_redirects=True,
        ) as client:
            response = await self._issue_request(client, params, headers)

            payload = response.json()
            items = payload.get("items") or []

            for raw in items:
                try:
                    records.append(MessageRecord.from_api(raw))
                except Exception:
                    # Skip malformed records but continue processing.
                    continue

        return records

    async def _issue_request(
        self,
        client: httpx.AsyncClient,
        params: dict[str, int],
        headers: dict[str, str],
    ) -> httpx.Response:
        try:
            response = await client.get("/messages", params=params, headers=headers)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            should_retry = status in {401, 405} and params.get("limit", 0) > 50
            if not should_retry:
                raise

            fallback_params = dict(params)
            fallback_params["limit"] = 50
            retry_response = await client.get(
                "/messages", params=fallback_params, headers=headers
            )
            retry_response.raise_for_status()
            return retry_response


__all__ = ["MessagesClient", "MessageRecord"]
