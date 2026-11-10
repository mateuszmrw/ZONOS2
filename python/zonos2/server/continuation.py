"""Bounded, process-local continuation cache for the single-worker proof of concept."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, replace
from typing import Callable

from pydantic import BaseModel, Field
from zonos2.message.tts import TTSContinuationContext, TTSTokenizeMsg


class ContinuationCacheConfig(BaseModel):
    ttl_seconds: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    max_entries: int = Field(default=32, ge=1)
    max_frames: int = Field(default=4096, ge=1)
    max_text_bytes: int = Field(default=16384, ge=1)

    @classmethod
    def from_env(cls) -> ContinuationCacheConfig:
        return cls.model_validate(
            {
                name: os.environ[f"ZONOS2_CONTINUATION_{name.upper()}"]
                for name in cls.model_fields
                if f"ZONOS2_CONTINUATION_{name.upper()}" in os.environ
            }
        )


class ContinuationUnavailable(ValueError):
    def __init__(self, message: str, status_code: int = 404) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class _Entry:
    owner: str
    template: TTSTokenizeMsg
    expires_at: float
    context: TTSContinuationContext | None = None
    error: str | None = None


class ContinuationCache:
    """Methods run on the API event loop, without awaits inside cache operations."""

    def __init__(
        self,
        config: ContinuationCacheConfig,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._entries: dict[str, _Entry] = {}

    def _prune(self) -> None:
        now = self._clock()
        self._entries = {
            key: entry for key, entry in self._entries.items() if entry.expires_at > now
        }

    def reserve(self, owner: str, template: TTSTokenizeMsg) -> str:
        self._prune()
        if len(self._entries) >= self.config.max_entries:
            # Never evict an active generation to admit another one.
            victim = next(
                (key for key, entry in self._entries.items() if entry.context or entry.error),
                None,
            )
            if victim is None:
                raise ContinuationUnavailable("Continuation cache is busy. Try again later.", 503)
            self._entries = {key: value for key, value in self._entries.items() if key != victim}
        key = uuid.uuid4().hex
        self._entries = {
            **self._entries,
            key: _Entry(
                owner=owner,
                template=replace(template, text="", continuation_context=None),
                expires_at=self._clock() + self.config.ttl_seconds,
            ),
        }
        return key

    def get(self, key: str, owner: str) -> TTSTokenizeMsg:
        self._prune()
        entry = self._entries.get(key)
        if entry is None or entry.owner != owner:
            raise ContinuationUnavailable("Continuation ID was not found or expired.")
        if entry.error:
            raise ContinuationUnavailable(entry.error, 409)
        if entry.context is None:
            raise ContinuationUnavailable(
                "Previous generation is not ready. Finish reading its response.", 409
            )
        return replace(entry.template, continuation_context=entry.context)

    def complete(
        self,
        uid: int,
        context: TTSContinuationContext | None,
        error: str | None = None,
    ) -> None:
        self._prune()
        if context is not None:
            frames = len(context.anchor_codes) + sum(map(len, context.history_codes))
            text_bytes = len(context.anchor_text.encode("utf-8")) + sum(
                len(text.encode("utf-8")) for text in context.history_text
            )
            if frames > self.config.max_frames or text_bytes > self.config.max_text_bytes:
                context = None
                error = "Continuation context exceeds the cache limit. Use smaller chunks."
        self._entries = {
            key: replace(
                entry,
                context=context,
                error=error if context is not None else (error or "No usable continuation context."),
                expires_at=self._clock() + self.config.ttl_seconds,
            )
            if entry.template.uid == uid
            else entry
            for key, entry in self._entries.items()
        }

    def abort(self, uid: int) -> None:
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.template.uid != uid or entry.context is not None or entry.error is not None
        }
