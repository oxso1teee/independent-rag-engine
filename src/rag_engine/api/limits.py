"""Request size limits and idempotency, kept apart from the routes that use them."""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send


class BodyTooLargeError(Exception):
    """Raised inside the request when its body exceeds the configured limit."""


class BodyLimitMiddleware:
    """Refuse bodies over ``max_bytes``: by ``Content-Length`` first, else by counting."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope["headers"]).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await self._refuse(scope, send)
            return
        seen = 0

        async def counting() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise BodyTooLargeError
            return message

        try:
            await self.app(scope, counting, send)
        except BodyTooLargeError:
            await self._refuse(scope, send)

    async def _refuse(self, scope: Scope, send: Send) -> None:
        body = (
            b'{"type":"urn:rag:problem:e-body-too-large","title":"Content Too Large","status":413,'
            b'"detail":"the request body exceeds the size limit","code":"E_BODY_TOO_LARGE",'
            b'"category":"user","retryable":false}'
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/problem+json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


@dataclass(frozen=True, slots=True)
class Stored:
    fingerprint: str
    status: int
    body: dict[str, Any]
    expires_at: float


class IdempotencyStore:
    """Remember the response of a POST under its ``Idempotency-Key`` for a while.

    The store lives in the memory of the server process: a retry within the TTL replays the first
    response, a restart forgets. The same key with a different body is a conflict, never a replay.
    """

    def __init__(self, ttl_s: float, capacity: int = 1024, clock: Any = time.monotonic) -> None:
        self._ttl = ttl_s
        self._capacity = capacity
        self._clock = clock
        self._items: OrderedDict[str, Stored] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def fingerprint(*parts: bytes) -> str:
        digest = hashlib.sha256()
        for part in parts:
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        return digest.hexdigest()

    def lookup(self, key: str, fingerprint: str) -> tuple[str, Stored | None]:
        """``("miss", None)``, ``("replay", stored)`` or ``("conflict", stored)``."""
        with self._lock:
            self._drop_expired()
            stored = self._items.get(key)
            if stored is None:
                return "miss", None
            return ("replay" if stored.fingerprint == fingerprint else "conflict"), stored

    def remember(self, key: str, fingerprint: str, status: int, body: dict[str, Any]) -> None:
        with self._lock:
            self._items[key] = Stored(fingerprint, status, body, self._clock() + self._ttl)
            self._items.move_to_end(key)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def _drop_expired(self) -> None:
        now = self._clock()
        for key in [k for k, v in self._items.items() if v.expires_at <= now]:
            del self._items[key]
