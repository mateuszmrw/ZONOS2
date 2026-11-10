"""CPU tests with real prompts, message serialization, and synthetic model/DAC output."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from zonos2.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTTSTokenizerMsg,
    TTSAudioReply,
    TTSDetokenizeMsg,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from zonos2.message.tts import TTSContinuationContext, TTSUserMsg
from zonos2.server import api_server
from zonos2.server.continuation import (
    ContinuationCache,
    ContinuationCacheConfig,
    ContinuationUnavailable,
)
from zonos2.tokenizer import server as tokenizer_server
from zonos2.tokenizer import vocoder
from zonos2.tts.longform import build_continuation_prompt
from zonos2.tts.prompt import TTSPromptBuilder, TTSPromptConfig


def _template(uid: int = 1) -> TTSTokenizeMsg:
    return TTSTokenizeMsg(uid=uid, text="First paragraph.", sampling_params=TTSSamplingParams())


def _context(value: int = 7) -> TTSContinuationContext:
    codes = [[value] * 9 for _ in range(4)]
    return TTSContinuationContext(1, "First paragraph.", codes, ["First paragraph."], [codes])


def _roundtrip_request(msg: TTSTokenizeMsg) -> TTSTokenizeMsg:
    decoded = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(cast(Any, msg)))
    assert isinstance(decoded, TTSTokenizeMsg)
    return decoded


def _roundtrip_reply(msg: TTSAudioReply) -> TTSAudioReply:
    decoded = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(cast(Any, msg)))
    assert isinstance(decoded, TTSAudioReply)
    return decoded


def test_cache_not_ready_ownership_expiry_and_abort() -> None:
    now = [0.0]
    cache = ContinuationCache(ContinuationCacheConfig(ttl_seconds=5), clock=lambda: now[0])
    key = cache.reserve("owner", _template())
    with pytest.raises(ContinuationUnavailable) as pending:
        cache.get(key, "owner")
    assert pending.value.status_code == 409
    cache.complete(1, _context())
    assert cache.get(key, "owner").continuation_context == _context()
    with pytest.raises(ContinuationUnavailable) as other:
        cache.get(key, "other")
    assert other.value.status_code == 404
    cache.abort(1)  # Finishing an HTTP response must not remove accepted context.
    assert cache.get(key, "owner").continuation_context is not None
    now[0] = 6
    with pytest.raises(ContinuationUnavailable):
        cache.get(key, "owner")
    aborted = cache.reserve("owner", _template(2))
    cache.abort(2)
    with pytest.raises(ContinuationUnavailable):
        cache.get(aborted, "owner")


def test_cache_capacity_and_frame_limit() -> None:
    cache = ContinuationCache(ContinuationCacheConfig(max_entries=1, max_frames=7))
    key = cache.reserve("owner", _template())
    with pytest.raises(ContinuationUnavailable) as busy:
        cache.reserve("owner", _template(2))
    assert busy.value.status_code == 503
    cache.complete(1, _context())  # Anchor + recent window exceeds seven frames.
    with pytest.raises(ContinuationUnavailable, match="cache limit"):
        cache.get(key, "owner")
    replacement = cache.reserve("owner", _template(2))
    assert replacement != key
    with pytest.raises(ContinuationUnavailable):
        cache.get(key, "owner")


def test_cache_configuration_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZONOS2_CONTINUATION_MAX_ENTRIES", "2")
    assert ContinuationCacheConfig.from_env().max_entries == 2
    monkeypatch.setenv("ZONOS2_CONTINUATION_TTL_SECONDS", "nan")
    with pytest.raises(ValidationError):
        ContinuationCacheConfig.from_env()


def test_context_survives_actual_message_decoders() -> None:
    request = replace(_template(), continuation_context=_context())
    decoded_request = _roundtrip_request(request)
    assert decoded_request.continuation_context == _context()
    reply = TTSAudioReply(1, b"", True, continuation_context=_context())
    decoded_reply = _roundtrip_reply(reply)
    assert decoded_reply.continuation_context == _context()


class _WorkerHarness:
    """Replace only transport, model sampling, and DAC weights, not worker logic."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pending: deque[Any] = deque()
        self.replies: list[TTSAudioReply] = []
        self.prompts: list[TTSUserMsg] = []
        self.eos_frame: int | None = 30
        monkeypatch.setenv("ZONOS2_TTS_NORM", "0")
        monkeypatch.setattr(tokenizer_server, "ZmqPullQueue", lambda *args, **kwargs: self)
        monkeypatch.setattr(
            tokenizer_server, "ZmqPushQueue", lambda addr, **kwargs: SimpleNamespace(
                put=self._backend if addr == "backend" else self._frontend
            )
        )
        monkeypatch.setattr(vocoder, "decode_dac", lambda codes: codes[:, :, 0].float().repeat_interleave(512, dim=1))

    def get(self) -> Any:
        if not self.pending:
            raise KeyboardInterrupt
        return self.pending.popleft()

    def empty(self) -> bool:
        return not self.pending

    def _frontend(self, reply: Any) -> None:
        self.replies.extend(reply.data if hasattr(reply, "data") else [reply])

    def _backend(self, msg: TTSUserMsg) -> None:
        self.prompts.append(msg)
        value = 100 + len(self.prompts)
        self.pending.append(BatchTTSTokenizerMsg([
            TTSDetokenizeMsg(
                msg.uid, [value] * 9, finished=i == 39,
                eos_frame=self.eos_frame if i == 39 else None,
            ) for i in range(40)
        ]))

    def run(self, msg: TTSTokenizeMsg, **worker_options: Any) -> list[TTSAudioReply]:
        self.replies = []
        self.pending.append(msg)
        tokenizer_server.tokenize_worker(
            addr="input", create=False, backend_addr="backend", frontend_addr="frontend",
            local_bs=1, text_vocab=519, **worker_options,
        )
        return self.replies


def test_worker_restores_anchor_and_recent_text_without_replaying_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    first = replace(_template(), retain_continuation=True)
    first_replies = worker.run(first)
    context = first_replies[-1].continuation_context
    assert context is not None
    assert len(context.history_codes[0]) == 27  # EOS at 30, minus boundary trim.

    second = replace(first, uid=2, text="Second paragraph.", continuation_context=context)
    second_replies = worker.run(second)
    next_context = second_replies[-1].continuation_context
    assert next_context is not None
    assert next_context.anchor_text == "First paragraph."
    assert next_context.history_text == ["Second paragraph."]

    third = replace(second, uid=3, text="Third paragraph.", continuation_context=next_context)
    third_replies = worker.run(third)
    expected = build_continuation_prompt(
        TTSPromptBuilder(TTSPromptConfig(text_vocab=519)),
        "First paragraph. Second paragraph.", "Third paragraph.",
        context.anchor_codes + next_context.history_codes[0], text_vocab=519,
    )
    assert torch.equal(worker.prompts[-1].input_ids, expected)
    for value, replies in zip((101, 102, 103), (first_replies, second_replies, third_replies)):
        pcm = np.frombuffer(b"".join(reply.audio_data for reply in replies), dtype=np.float32)
        assert len(pcm) == 27 * 512
        np.testing.assert_allclose(pcm, value, rtol=0, atol=2e-5)


def test_worker_rejects_oversized_prompt_and_incomplete_context(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    replies = worker.run(replace(_template(), retain_continuation=True, max_prompt_frames=1))
    assert replies[-1].finished
    assert "model context" in str(replies[-1].error)
    assert not worker.prompts
    worker.eos_frame = None
    replies = worker.run(replace(_template(), retain_continuation=True))
    assert replies[-1].continuation_context is None
    assert replies[-1].continuation_error


def test_prompt_budget_includes_speaker_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    request = replace(_template(), retain_continuation=True, speaker_embedding=torch.ones(3))
    options = {"speaker_background_num_buckets": 2, "accurate_mode_num_buckets": 1}
    worker.run(request, **options)
    prompt_frames = len(worker.prompts[-1].input_ids)
    replies = worker.run(replace(request, uid=2, max_prompt_frames=prompt_frames), **options)
    assert replies[-1].error
    assert len(worker.prompts) == 1


def test_continuation_reuses_prefix_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    request = replace(
        _template(), text="Next paragraph.", retain_continuation=True,
        continuation_context=_context(), sampling_params=TTSSamplingParams(prefix_cfg_scale=1.2),
    )
    worker.run(request)
    fresh = TTSPromptBuilder(TTSPromptConfig(text_vocab=519)).build(request.text)
    assert torch.equal(worker.prompts[-1].cfg_uncond_input_ids, fresh)


def _frontend(monkeypatch: pytest.MonkeyPatch, worker: _WorkerHarness) -> api_server.FrontendManager:
    incoming: asyncio.Queue[TTSAudioReply] = asyncio.Queue()

    async def send(msg: TTSTokenizeMsg) -> None:
        # Exercise the real cross-process message codecs as well as the worker.
        decoded = _roundtrip_request(msg)
        for reply in worker.run(decoded):
            await incoming.put(_roundtrip_reply(reply))

    config: Any = SimpleNamespace(
        num_tokenizer=0, max_seq_len=2048,
        model_config=SimpleNamespace(speaker_enabled=False),
    )
    sender: Any = SimpleNamespace(put=send, stop=lambda: None)
    receiver: Any = SimpleNamespace(get=incoming.get, stop=lambda: None)
    manager = api_server.FrontendManager(config, sender, receiver)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", manager)
    return manager


def test_http_chaining_inherits_speaker_and_supports_final_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    speaker = torch.tensor([0.1, 0.2, 0.3])

    async def resolve_speaker(**kwargs: Any) -> torch.Tensor:
        return speaker

    monkeypatch.setattr(api_server, "_resolve_speaker_embedding", resolve_speaker)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        first = await api_server.tts_generate(
            api_server.TTSGenerateRequest(
                text="First paragraph.", retain_continuation=True, stream=False, temperature=0.8,
            ), "owner",
        )
        key = first.headers["X-TTS-Continuation-ID"]
        assert state.continuations.get(key, "owner").speaker_embedding is not None
        second = await api_server.tts_generate(
            api_server.TTSGenerateRequest(text="Second paragraph.", continuation_of=key, stream=False),
            "owner",
        )
        next_key = second.headers["X-TTS-Continuation-ID"]
        assert next_key != key
        assert worker.prompts[-1].sampling_params.temperature == 0.8
        assert torch.equal(worker.prompts[-1].speaker_embedding, speaker)
        last = await api_server.tts_generate(
            api_server.TTSGenerateRequest(
                text="Third paragraph.", continuation_of=next_key, retain_continuation=False, stream=True,
            ), "owner",
        )
        assert "X-TTS-Continuation-ID" not in last.headers
        pcm = b"".join([chunk async for chunk in last.body_iterator])
        assert len(pcm) == 30 * 512 * 4  # Final request keeps its normal ending.
        assert not state.ack_map and not state.event_map and not state.start_time_map

    asyncio.run(scenario())


@pytest.mark.parametrize("owner, expected", [(None, 400), ("owner", 404)])
def test_http_missing_session_and_unknown_id(monkeypatch: pytest.MonkeyPatch, owner: str | None, expected: int) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        _frontend(monkeypatch, worker)
        with pytest.raises(HTTPException) as error:
            await api_server.tts_generate(
                api_server.TTSGenerateRequest(text="Next.", continuation_of="unknown"), owner,
            )
        assert error.value.status_code == expected
        assert not worker.prompts

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
def test_http_route_returns_custom_header_and_pcm(monkeypatch: pytest.MonkeyPatch, stream: bool) -> None:
    worker = _WorkerHarness(monkeypatch)
    _frontend(monkeypatch, worker)
    with TestClient(api_server.app) as client:
        response = client.post(
            "/tts/generate", headers={"X-TTS-Session-ID": "owner"},
            json={"text": "First paragraph.", "retain_continuation": True, "stream": stream},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == 27 * 512 * 4
        last = client.post(
            "/tts/generate", headers={"X-TTS-Session-ID": "owner"},
            json={
                "text": "The ending.", "stream": stream,
                "continuation_of": response.headers["X-TTS-Continuation-ID"],
                "retain_continuation": False,
            },
        )
        assert last.status_code == 200
        assert "X-TTS-Continuation-ID" not in last.headers
        assert len(last.content) == 30 * 512 * 4


def test_http_checks_owner_pending_settings_and_tokenizer_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        key = state.continuations.reserve("owner", _template())

        async def rejected(owner: str, status: int, **kwargs: Any) -> None:
            with pytest.raises(HTTPException) as error:
                await api_server.tts_generate(
                    api_server.TTSGenerateRequest(text="Next.", continuation_of=key, **kwargs), owner,
                )
            assert error.value.status_code == status

        await rejected("other", 404)
        await rejected("owner", 409)
        state.continuations.complete(1, _context())
        await rejected("owner", 400, language="de")
        state.config.num_tokenizer = 1
        await rejected("owner", 400)
        assert not worker.prompts

    asyncio.run(scenario())


def test_http_reports_prompt_error_before_opening_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        state.config.max_seq_len = 2
        with pytest.raises(HTTPException, match="model context"):
            await api_server.tts_generate(
                api_server.TTSGenerateRequest(text="First.", retain_continuation=True), "owner",
            )
        assert not state.ack_map and not state.event_map and not state.start_time_map

    asyncio.run(scenario())


def test_failed_submission_removes_pending_context(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)

        async def fail(msg: Any) -> None:
            raise RuntimeError("Synthetic transport failure")

        state.send_tokenizer.put = fail
        with pytest.raises(RuntimeError, match="Synthetic transport failure"):
            await api_server.tts_generate(
                api_server.TTSGenerateRequest(text="First.", retain_continuation=True), "owner",
            )
        assert not state.ack_map and not state.event_map and not state.start_time_map
        assert not state.continuations._entries

    asyncio.run(scenario())
