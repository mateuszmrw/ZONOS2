"""CPU tests with real prompts, message serialization, and synthetic model/DAC output."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import msgpack  # type: ignore[import-untyped]
import numpy as np
import pytest
import torch
from fastapi import HTTPException
from fastapi.responses import Response
from fastapi.testclient import TestClient
from httpx import Response as HTTPResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect
from starlette.types import Message
from zonos2.core import TTSReq
from zonos2.core import TTSSamplingParams as RuntimeSamplingParams
from zonos2.message import (
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTTSTokenizerMsg,
    TTSAudioReply,
    TTSDetokenizeMsg,
    TTSSamplingParams,
    TTSTokenizeMsg,
)
from zonos2.message.tokenizer import AbortMsg
from zonos2.message.tts import (
    TTSAudioTail,
    TTSCancelMsg,
    TTSContinuationContext,
    TTSFinishAudioMsg,
    TTSFinishReason,
    TTSUserMsg,
)
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
    decoded: object = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(cast(Any, msg)))
    assert isinstance(decoded, TTSTokenizeMsg)
    return decoded


def _roundtrip_reply(msg: TTSAudioReply) -> TTSAudioReply:
    decoded: object = BaseFrontendMsg.decoder(
        msgpack.unpackb(msgpack.packb(BaseFrontendMsg.encoder(cast(Any, msg))), raw=False)
    )
    assert isinstance(decoded, TTSAudioReply)
    return decoded


def test_cache_not_ready_ownership_expiry_and_abort() -> None:
    now = [0.0]
    cache = ContinuationCache(ContinuationCacheConfig(ttl_seconds=5), clock=lambda: now[0])
    key = cache.reserve("owner", _template())
    with pytest.raises(ContinuationUnavailable) as pending:
        cache.get(key, "owner")
    assert pending.value.status_code == 409
    cache.complete(1, _context(), finish_reason="complete")
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
    cache.complete(1, _context(), finish_reason="complete")  # Anchor + recent exceeds seven frames.
    with pytest.raises(ContinuationUnavailable, match="cache limit"):
        cache.get(key, "owner")
    replacement = cache.reserve("owner", _template(2))
    assert replacement != key
    with pytest.raises(ContinuationUnavailable):
        cache.get(key, "owner")


def test_audio_tail_cache_protects_active_ids_without_extending_expiry() -> None:
    now = [0.0]
    cache = ContinuationCache(
        ContinuationCacheConfig(ttl_seconds=5, max_entries=1), clock=lambda: now[0],
    )
    key = cache.reserve("owner", _template(), retain_model=False, track_audio=True)
    with pytest.raises(ContinuationUnavailable) as pending:
        cache.hold_audio_tail(2, key, "owner")
    assert pending.value.status_code == 409
    tail = TTSAudioTail([[1] * 9], 0, b"", [[2] * 9 for _ in range(8)])
    cache.complete(1, None, finish_reason="complete", audio_tail=tail)
    now[0] = 4.0
    assert cache.hold_audio_tail(2, key, "owner") == tail
    with pytest.raises(ContinuationUnavailable) as busy:
        cache.reserve("other", _template(3))
    assert busy.value.status_code == 503
    now[0] = 6.0
    assert cache.reserve("other", _template(3)) != key
    with pytest.raises(ContinuationUnavailable) as expired:
        cache.audio_status(key, "owner")
    assert expired.value.status_code == 404


def test_discard_pending_tail_before_late_worker_reply() -> None:
    cache = ContinuationCache(ContinuationCacheConfig())
    key = cache.reserve("owner", _template(), track_audio=True)
    cache.release_audio_tail(key, "owner")
    tail = TTSAudioTail([[1] * 9], 0, b"", [[2] * 9 for _ in range(8)])
    cache.complete(1, _context(), finish_reason="complete", audio_tail=tail)
    assert cache.status(key, "owner").usable
    with pytest.raises(ContinuationUnavailable) as discarded:
        cache.hold_audio_tail(2, key, "owner")
    assert discarded.value.status_code == 404


def test_cache_does_not_mark_empty_context_usable() -> None:
    cache = ContinuationCache(ContinuationCacheConfig())
    key = cache.reserve("owner", _template())
    cache.complete(1, TTSContinuationContext(1, "", [], [], []), finish_reason="complete")
    status = cache.status(key, "owner")
    assert status.finish_reason == "complete"
    assert not status.usable
    with pytest.raises(ContinuationUnavailable) as unavailable:
        cache.get(key, "owner")
    assert unavailable.value.status_code == 409


def test_cache_protects_parent_until_both_children_exit() -> None:
    cache = ContinuationCache(ContinuationCacheConfig(max_entries=2))
    parent = cache.reserve("owner", _template())
    cache.complete(1, _context(), finish_reason="complete")
    cache.hold_parent(2, parent, "owner")
    cache.hold_parent(3, parent, "owner")
    child = cache.reserve("owner", _template(2))
    with pytest.raises(ContinuationUnavailable) as busy:
        cache.reserve("other", _template(4))
    assert busy.value.status_code == 503
    cache.complete(2, None, finish_reason="frame_limit")
    cache.abort(2)
    cache.reserve("other", _template(4))
    assert cache.status(parent, "owner").usable
    with pytest.raises(ContinuationUnavailable) as evicted_child:
        cache.status(child, "owner")
    assert evicted_child.value.status_code == 404
    cache.abort(3)
    cache.reserve("other", _template(5))
    with pytest.raises(ContinuationUnavailable) as evicted_parent:
        cache.status(parent, "owner")
    assert evicted_parent.value.status_code == 404


def test_cache_parent_hold_does_not_extend_expiry() -> None:
    now = [0.0]
    cache = ContinuationCache(ContinuationCacheConfig(ttl_seconds=5), clock=lambda: now[0])
    parent = cache.reserve("owner", _template())
    cache.complete(1, _context(), finish_reason="complete")
    now[0] = 4.0
    cache.hold_parent(2, parent, "owner")
    child = cache.reserve("owner", _template(2))
    now[0] = 6.0
    with pytest.raises(ContinuationUnavailable) as expired:
        cache.get(parent, "owner")
    assert expired.value.status_code == 404
    assert cache.status(child, "owner").status == "pending"


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
    reply = TTSAudioReply(1, b"", True, continuation_context=_context(), finish_reason="complete")
    decoded_reply = _roundtrip_reply(reply)
    assert decoded_reply.continuation_context == _context()
    assert decoded_reply.finish_reason == "complete"


@pytest.mark.parametrize("codebook", range(9))
@pytest.mark.parametrize("capped", [False, True])
def test_scheduler_reason_requires_eos_drain(codebook: int, capped: bool) -> None:
    request = TTSReq(
        torch.zeros((2, 10), dtype=torch.int32),
        table_idx=0, cached_len=0, output_len=17 if capped else 25,
        uid=0, sampling_params=RuntimeSamplingParams(), cache_handle=cast(Any, None),
    )
    reason = None
    for index in range(request.output_len):
        request.complete_one()
        frame = [10] * 9
        if index == 15:
            frame[codebook] = request.eoa_id
        reason = request.update_finish_reason(frame, max_seq_len=200)
        if reason is not None:
            break
    assert request.eos_frame == 15 - codebook
    assert reason == ("frame_limit" if capped else "complete")
    assert (request.eos_countdown > 0) == capped


class _WorkerHarness:
    """Replace only transport, model sampling, and DAC weights, not worker logic."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.pending: deque[Any] = deque()
        self.replies: list[TTSAudioReply] = []
        self.prompts: list[TTSUserMsg] = []
        self.eos_frame: int | None = 30
        self.finish_reason: TTSFinishReason = "complete"
        monkeypatch.setenv("ZONOS2_TTS_NORM", "0")
        monkeypatch.setattr(tokenizer_server, "ZmqPullQueue", lambda *args, **kwargs: self)
        monkeypatch.setattr(
            tokenizer_server, "ZmqPushQueue", lambda addr, **kwargs: SimpleNamespace(
                put=self._backend if addr == "backend" else self._frontend
            )
        )
        monkeypatch.setattr(
            vocoder,
            "decode_dac",
            lambda codes: codes[:, :, 0].float().repeat_interleave(512, dim=1).cpu(),
        )

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
                finish_reason=self.finish_reason if i == 39 else None,
            ) for i in range(40)
        ]))

    def run(
        self, msg: TTSTokenizeMsg | TTSFinishAudioMsg, **worker_options: Any,
    ) -> list[TTSAudioReply]:
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


@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("linked", [False, True])
def test_worker_decodes_all_nine_codebooks_at_request_end(
    monkeypatch: pytest.MonkeyPatch, retain: bool, linked: bool
) -> None:
    worker = _WorkerHarness(monkeypatch)
    eos, codebooks = 52, 9
    aligned = [[20 + i * codebooks + j for j in range(codebooks)] for i in range(eos)]
    raw = [
        [aligned[i - j][j] if 0 <= i - j < eos else 1025 for j in range(codebooks)]
        for i in range(eos + codebooks)
    ]

    def send_frames(msg: TTSUserMsg) -> None:
        worker.pending.append(BatchTTSTokenizerMsg([
            TTSDetokenizeMsg(
                uid=msg.uid, audio_codes=row, finished=i == len(raw) - 1,
                eos_frame=eos if i >= eos else None,
                finish_reason="complete" if i == len(raw) - 1 else None,
            )
            for i, row in enumerate(raw)
        ]))

    decoded: set[tuple[int, ...]] = set()

    def capture_dac(codes: torch.Tensor) -> torch.Tensor:
        decoded.update(tuple(row) for row in codes[0].cpu().tolist() if row[0] >= 20)
        return torch.zeros((1, codes.shape[1] * 512), dtype=torch.float32)

    monkeypatch.setattr(worker, "_backend", send_frames)
    monkeypatch.setattr(vocoder, "decode_dac", capture_dac)
    replies = worker.run(replace(
        _template(), long_form=False, retain_continuation=retain,
        continuation_context=_context() if linked else None,
    ))
    kept = eos - 3 if retain else eos
    assert replies[-1].finish_reason == "complete"
    assert sum(len(reply.audio_data) for reply in replies) == kept * 512 * 4
    context = replies[-1].continuation_context
    if retain:
        assert context is not None
        assert context.history_codes == [raw[:kept]]
    else:
        assert context is None
    assert decoded == {tuple(row) for row in aligned[:kept]}


def test_worker_rejects_oversized_prompt_and_incomplete_context(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    replies = worker.run(replace(_template(), retain_continuation=True, max_prompt_frames=1))
    assert replies[-1].finished
    assert "model context" in str(replies[-1].error)
    assert not worker.prompts
    worker.eos_frame = None
    worker.finish_reason = "frame_limit"
    replies = worker.run(replace(_template(), retain_continuation=True))
    assert replies[-1].continuation_context is None
    assert replies[-1].continuation_error
    assert replies[-1].finish_reason == "frame_limit"


def test_worker_stops_long_form_at_first_incomplete_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    worker.finish_reason = "frame_limit"  # EOS seen, but delayed codebooks did not finish.
    result = worker.run(replace(
        _template(), text="The first sentence. The second sentence.", retain_continuation=True,
        long_form=True, long_form_chunk_chars=20, long_form_split_mode="sentence",
    ))
    assert len(worker.prompts) == 1
    assert result[-1].finished
    assert result[-1].finish_reason == "frame_limit"
    assert result[-1].continuation_context is None
    assert len(b"".join(reply.audio_data for reply in result)) == 30 * 512 * 4


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
    unconditioned = worker.prompts[-1].cfg_uncond_input_ids
    assert unconditioned is not None
    assert torch.equal(unconditioned, fresh)


def _frontend(
    monkeypatch: pytest.MonkeyPatch,
    worker: _WorkerHarness,
    incoming: asyncio.Queue[TTSAudioReply] | None = None,
) -> api_server.FrontendManager:
    inbox: asyncio.Queue[TTSAudioReply] = incoming if incoming is not None else asyncio.Queue()

    async def send(msg: TTSTokenizeMsg | TTSFinishAudioMsg) -> None:
        # Exercise the real cross-process message codecs as well as the worker.
        wire = msgpack.packb(BaseTokenizerMsg.encoder(cast(Any, msg)), use_bin_type=True)
        decoded: object = BaseTokenizerMsg.decoder(msgpack.unpackb(wire, raw=False))
        assert isinstance(decoded, (TTSTokenizeMsg, TTSFinishAudioMsg))
        for reply in worker.run(decoded):
            await inbox.put(_roundtrip_reply(reply))

    config: Any = SimpleNamespace(
        num_tokenizer=0, max_seq_len=2048,
        model_config=SimpleNamespace(speaker_enabled=False),
    )
    sender: Any = SimpleNamespace(put=send, stop=lambda: None)
    receiver: Any = SimpleNamespace(get=inbox.get, stop=lambda: None)
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
        embedding = worker.prompts[-1].speaker_embedding
        assert embedding is not None
        assert torch.equal(embedding, speaker)
        last = await api_server.tts_generate(
            api_server.TTSGenerateRequest(
                text="Third paragraph.", continuation_of=next_key, retain_continuation=False, stream=True,
            ), "owner",
        )
        assert "X-TTS-Continuation-ID" not in last.headers
        body_parts: list[bytes] = []

        async def receive() -> Message:
            return await asyncio.Future[Message]()

        async def send(message: Message) -> None:
            if message["type"] == "http.response.body":
                body_parts.append(message["body"])

        await last({"type": "http", "asgi": {"spec_version": "2.3"}}, receive, send)
        pcm = b"".join(body_parts)
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


@pytest.mark.parametrize("stream", [False, True])
def test_audio_track_order_is_separate_from_model_branches(
    monkeypatch: pytest.MonkeyPatch, stream: bool,
) -> None:
    worker = _WorkerHarness(monkeypatch)
    manager = _frontend(monkeypatch, worker)
    owner = {"X-TTS-Session-ID": "owner"}
    with TestClient(api_server.app) as client:
        def generate(text: str, **options: Any) -> HTTPResponse:
            response = client.post(
                "/tts/generate", headers=owner, json={"text": text, "stream": stream, **options},
            )
            assert response.status_code == 200
            return response

        a = generate("First.", retain_continuation=True, defer_audio_tail=True)
        anchor = a.headers["X-TTS-Continuation-ID"]
        a_tail = a.headers["X-TTS-Audio-Tail-ID"]
        assert a.content
        bad_owner = {"X-TTS-Session-ID": "wrong"}
        status_path = a.headers["X-TTS-Audio-Status-URL"]
        assert client.get(status_path, headers=bad_owner).status_code == 404
        assert client.post(status_path + "/finish", headers=bad_owner).status_code == 404
        assert client.delete(status_path, headers=bad_owner).status_code == 404
        assert client.get(status_path, headers=owner).json() == {
            "finish_reason": "complete", "audio_tail_pending": True, "usable": True,
        }
        c = generate(
            "After first.", continuation_of=anchor, audio_tail_of=a_tail,
            retain_continuation=False, defer_audio_tail=True,
        )
        assert "X-TTS-Continuation-ID" not in c.headers
        assert client.get(c.headers["X-TTS-Audio-Status-URL"], headers=owner).json() == {
            "finish_reason": "complete", "audio_tail_pending": True, "usable": False,
        }
        assert client.delete(status_path, headers=owner).status_code == 204
        assert client.get(f"/tts/continuations/{anchor}", headers=owner).json()["usable"]
        b = generate(
            "Branch from first.", continuation_of=anchor,
            audio_tail_of=c.headers["X-TTS-Audio-Tail-ID"], defer_audio_tail=True,
        )
        d = generate(
            "Guten Tag.", language="de", retain_continuation=True,
            audio_tail_of=b.headers["X-TTS-Audio-Tail-ID"], defer_audio_tail=True,
        )
        e = generate(
            "Branch again.", continuation_of=anchor,
            audio_tail_of=d.headers["X-TTS-Audio-Tail-ID"], defer_audio_tail=True,
        )
        final_status = e.headers["X-TTS-Audio-Status-URL"]
        ending = client.post(final_status + "/finish", headers=owner)
        assert ending.status_code == 200 and ending.headers["X-TTS-Generation-Status"] == "complete"
        assert client.post(final_status + "/finish", headers=owner).content == ending.content
        assert client.delete(final_status, headers=owner).status_code == 204
        assert client.post(final_status + "/finish", headers=owner).status_code == 404
        combined = np.frombuffer(
            b"".join(part.content for part in (a, c, b, d, e, ending)), dtype=np.float32,
        )
        assert combined.size == 5 * 27 * 512
        # The fake decoder is constant per phrase, so each must occupy its exact slot.
        expected = np.repeat(np.arange(101, 106, dtype=np.float32), 27 * 512)
        np.testing.assert_allclose(combined, expected, rtol=0, atol=2e-5)
        assert len(worker.prompts) == 5
        assert not manager.active_workers


def test_failed_track_attempt_can_retry_the_same_audio_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    _frontend(monkeypatch, worker)
    headers = {"X-TTS-Session-ID": "owner"}
    with TestClient(api_server.app) as client:
        a = client.post("/tts/generate", headers=headers, json={
            "text": "First.", "stream": True, "defer_audio_tail": True,
        })
        body = {"text": "Second.", "stream": True, "audio_tail_of": a.headers["X-TTS-Audio-Tail-ID"]}
        worker.finish_reason = "frame_limit"
        failed = client.post("/tts/generate", headers=headers, json={**body, "defer_audio_tail": True})
        assert client.get(failed.headers["X-TTS-Audio-Status-URL"], headers=headers).json() == {
            "finish_reason": "frame_limit", "audio_tail_pending": False, "usable": False,
        }
        assert client.post(
            failed.headers["X-TTS-Audio-Status-URL"] + "/finish", headers=headers,
        ).status_code == 409
        worker.finish_reason = "complete"
        b = client.post("/tts/generate", headers=headers, json=body)
        assert client.get(b.headers["X-TTS-Audio-Status-URL"], headers=headers).json() == {
            "finish_reason": "complete", "audio_tail_pending": False, "usable": False,
        }
        assert len(a.content + b.content) == (27 + 30) * 512 * 4


def test_http_rejects_retention_without_any_context(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    _frontend(monkeypatch, worker)
    body = {
        "text": "First.", "stream": False,
        "long_form_window_chunks": 1, "long_form_pin_anchor": False,
    }
    with TestClient(api_server.app) as client:
        retained = client.post(
            "/tts/generate", headers={"X-TTS-Session-ID": "owner"},
            json={**body, "retain_continuation": True},
        )
        assert retained.status_code == 400
        assert not worker.prompts
        assert client.post("/tts/generate", json=body).status_code == 200
        pinned = client.post(
            "/tts/generate", headers={"X-TTS-Session-ID": "owner"},
            json={**body, "long_form_pin_anchor": True, "retain_continuation": True},
        )
        assert pinned.status_code == 200
        status = client.get(
            pinned.headers["X-TTS-Continuation-Status-URL"],
            headers={"X-TTS-Session-ID": "owner"},
        )
        assert status.json()["usable"] is True


def test_http_one_slot_allows_only_a_nonretained_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZONOS2_CONTINUATION_MAX_ENTRIES", "1")
    worker = _WorkerHarness(monkeypatch)
    _frontend(monkeypatch, worker)
    headers = {"X-TTS-Session-ID": "owner"}
    with TestClient(api_server.app) as client:
        first = client.post(
            "/tts/generate", headers=headers,
            json={"text": "First.", "retain_continuation": True, "stream": False},
        )
        parent = first.headers["X-TTS-Continuation-ID"]
        body = {"text": "Next.", "continuation_of": parent, "stream": False}
        blocked = client.post("/tts/generate", headers=headers, json=body)
        assert blocked.status_code == 503
        assert len(worker.prompts) == 1
        last = client.post(
            "/tts/generate", headers=headers, json={**body, "retain_continuation": False},
        )
        assert last.headers["X-TTS-Generation-Status"] == "complete"
        assert "X-TTS-Continuation-ID" not in last.headers
        status = client.get(first.headers["X-TTS-Continuation-Status-URL"], headers=headers)
        assert status.json()["usable"] is True


def test_http_can_branch_from_a_ready_id_and_continue_a_child(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZONOS2_CONTINUATION_MAX_ENTRIES", "3")
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        first = await api_server.tts_generate(
            api_server.TTSGenerateRequest(text="First.", retain_continuation=True, stream=False),
            "owner",
        )
        parent = first.headers["X-TTS-Continuation-ID"]
        sent: asyncio.Queue[BaseTokenizerMsg] = asyncio.Queue()
        submit = state.send_tokenizer.put

        async def start(text: str) -> Response:
            result = await api_server.tts_generate(
                api_server.TTSGenerateRequest(
                    text=text, continuation_of=parent, retain_continuation=True, stream=True,
                ),
                "owner",
            )
            assert isinstance(result, Response)
            return result

        with monkeypatch.context() as mp:
            mp.setattr(state.send_tokenizer, "put", sent.put)
            left, right = await asyncio.gather(start("Left."), start("Right."))
            left_id = left.headers["X-TTS-Continuation-ID"]
            right_id = right.headers["X-TTS-Continuation-ID"]
            assert len({parent, left_id, right_id}) == 3
            assert state.continuations.status(left_id, "owner").status == "pending"
            assert state.continuations.status(right_id, "owner").status == "pending"
            with pytest.raises(HTTPException) as early:
                await api_server.tts_generate(
                    api_server.TTSGenerateRequest(text="Too soon.", continuation_of=left_id),
                    "owner",
                )
            assert early.value.status_code == 409
            with pytest.raises(HTTPException) as busy:
                await api_server.tts_generate(
                    api_server.TTSGenerateRequest(text="Unrelated.", retain_continuation=True),
                    "other",
                )
            assert busy.value.status_code == 503
            assert sent.qsize() == 2

        async def receive() -> Message:
            return await asyncio.Future[Message]()

        async def send(_: Message) -> None:
            pass

        await submit(sent.get_nowait())
        await submit(sent.get_nowait())
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await asyncio.wait_for(
            asyncio.gather(left(scope, receive, send), right(scope, receive, send)), timeout=2,
        )
        assert state.continuations.status(parent, "owner").usable
        for key, text in ((left_id, "Left."), (right_id, "Right.")):
            context = state.continuations.get(key, "owner").continuation_context
            assert context is not None
            assert context.anchor_text == "First."
            assert context.history_text == [text]

        next_step = await api_server.tts_generate(
            api_server.TTSGenerateRequest(
                text="Third.", continuation_of=left_id, retain_continuation=True, stream=False,
            ),
            "owner",
        )
        assert next_step.headers["X-TTS-Generation-Status"] == "complete"
        key = next_step.headers["X-TTS-Continuation-ID"]
        assert key not in {parent, left_id, right_id}
        assert state.continuations.status(key, "owner").usable
        assert state.continuations.status(right_id, "owner").usable

    asyncio.run(scenario())


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("early_eos", [None, 30])
def test_http_reports_truncation_and_retries_from_last_good_id(
    monkeypatch: pytest.MonkeyPatch, stream: bool, early_eos: int | None
) -> None:
    monkeypatch.setenv("ZONOS2_CONTINUATION_MAX_ENTRIES", "2")
    worker = _WorkerHarness(monkeypatch)
    _frontend(monkeypatch, worker)
    headers = {"X-TTS-Session-ID": "owner"}
    with TestClient(api_server.app) as client:
        first = client.post(
            "/tts/generate",
            headers=headers,
            json={"text": "First paragraph.", "stream": False, "retain_continuation": True},
        )
        assert first.headers["X-TTS-Generation-Status"] == "complete"
        parent = first.headers["X-TTS-Continuation-ID"]
        other_headers = {"X-TTS-Session-ID": "other"}
        unrelated = client.post(
            "/tts/generate", headers=other_headers,
            json={"text": "Unrelated.", "retain_continuation": True, "stream": False},
        )
        assert unrelated.headers["X-TTS-Generation-Status"] == "complete"
        worker.eos_frame = early_eos
        worker.finish_reason = "frame_limit"
        failed = client.post(
            "/tts/generate",
            headers=headers,
            json={"text": "This request cannot finish.", "continuation_of": parent, "stream": stream},
        )
        assert failed.status_code == 200
        assert len(failed.content) == (32 if early_eos is None else 30) * 512 * 4
        if not stream:
            assert failed.headers["X-TTS-Generation-Status"] == "frame_limit"
        status_url = failed.headers["X-TTS-Continuation-Status-URL"]
        status = client.get(status_url, headers=headers)
        assert status.headers["Cache-Control"] == "no-store"
        assert status.json()["finish_reason"] == "frame_limit"
        assert status.json()["usable"] is False
        assert client.get(status_url, headers={"X-TTS-Session-ID": "other"}).status_code == 404
        assert client.get(
            unrelated.headers["X-TTS-Continuation-Status-URL"], headers=other_headers,
        ).status_code == 404
        rejected = client.post(
            "/tts/generate",
            headers=headers,
            json={"text": "Next.", "continuation_of": failed.headers["X-TTS-Continuation-ID"]},
        )
        assert rejected.status_code == 409
        worker.eos_frame = 30
        worker.finish_reason = "complete"
        retry = client.post(
            "/tts/generate",
            headers=headers,
            json={
                "text": "A shorter retry.", "continuation_of": parent,
                "retain_continuation": False, "stream": False,
            },
        )
        assert retry.headers["X-TTS-Generation-Status"] == "complete"
        pcm = np.frombuffer(retry.content, dtype=np.float32)
        assert len(pcm) == 30 * 512
        np.testing.assert_allclose(pcm, 104.0, rtol=0, atol=2e-5)


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
        state.continuations.complete(1, _context(), finish_reason="complete")
        await rejected("owner", 400, language="de")
        monkeypatch.setattr(state.config, "num_tokenizer", 1)
        await rejected("owner", 400)
        assert not worker.prompts

    asyncio.run(scenario())


def test_http_reports_prompt_error_before_opening_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)
    state = _frontend(monkeypatch, worker)
    monkeypatch.setattr(state.config, "max_seq_len", 2)
    with TestClient(api_server.app) as client:
        response = client.post(
            "/tts/generate",
            headers={"X-TTS-Session-ID": "owner"},
            json={"text": "First.", "retain_continuation": True, "stream": True},
        )
    assert response.status_code == 400
    assert "model context" in response.json()["detail"]
    assert not state.ack_map and not state.event_map and not state.start_time_map


@pytest.mark.parametrize(
    "spec, disconnect_at",
    [
        ("2.3", "waiting_for_audio"),
        ("2.3", "http.response.start"),
        ("2.3", "http.response.body"),
        ("2.4", "http.response.start"),
        ("2.4", "http.response.body"),
        ("2.4", "waiting_for_audio"),
    ],
)
def test_stream_disconnect_cleans_request_and_preserves_parent(
    monkeypatch: pytest.MonkeyPatch, spec: str, disconnect_at: str
) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        parent = await api_server.tts_generate(
            api_server.TTSGenerateRequest(text="First.", stream=False, retain_continuation=True),
            "owner",
        )
        parent_id = parent.headers["X-TTS-Continuation-ID"]
        submitted: asyncio.Queue[TTSTokenizeMsg | AbortMsg] = asyncio.Queue()
        waiting = asyncio.Event()
        disconnected = asyncio.Event()
        wait_for_tts_start = state.wait_for_tts_start

        async def observe_start(uid: int) -> None:
            waiting.set()
            await wait_for_tts_start(uid)

        with monkeypatch.context() as mp:
            mp.setattr(state.send_tokenizer, "put", submitted.put)
            mp.setattr(state, "wait_for_tts_start", observe_start)
            response_task = asyncio.create_task(api_server.tts_generate(
                api_server.TTSGenerateRequest(text="Next.", continuation_of=parent_id, stream=True),
                "owner",
            ))
            msg = await asyncio.wait_for(submitted.get(), timeout=2)
            assert isinstance(msg, TTSTokenizeMsg)
            if disconnect_at != "waiting_for_audio":
                state.ack_map[msg.uid].append(TTSAudioReply(msg.uid, b"\0" * 4, False))
                state.event_map[msg.uid].set()
            response = await asyncio.wait_for(response_task, timeout=2)
            child_id = response.headers["X-TTS-Continuation-ID"]

            async def receive() -> Message:
                when = waiting if disconnect_at == "waiting_for_audio" else disconnected
                await when.wait()
                return {"type": "http.disconnect"}

            async def send(message: Message) -> None:
                if message["type"] != disconnect_at:
                    return
                if spec == "2.4":
                    raise OSError("Client closed the connection")
                disconnected.set()
                if disconnect_at == "http.response.start":
                    # Hold the header send open so the iterator cannot start.
                    await asyncio.Future[None]()

            exchange = response({"type": "http", "asgi": {"spec_version": spec}}, receive, send)
            if spec == "2.4" and disconnect_at != "waiting_for_audio":
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(exchange, timeout=2)
            else:
                await asyncio.wait_for(exchange, timeout=2)
            assert await asyncio.wait_for(submitted.get(), timeout=2) == AbortMsg(msg.uid)

        assert not state.ack_map and not state.event_map and not state.start_time_map
        assert not state.active_workers
        assert not state.continuations._active_parents
        with pytest.raises(ContinuationUnavailable) as gone:
            state.continuations.get(child_id, "owner")
        assert gone.value.status_code == 404
        assert state.continuations.status(parent_id, "owner").usable
        retry = await api_server.tts_generate(
            api_server.TTSGenerateRequest(
                text="Try again.", continuation_of=parent_id,
                retain_continuation=False, stream=False,
            ),
            "owner",
        )
        assert retry.headers["X-TTS-Generation-Status"] == "complete"

    asyncio.run(scenario())


def test_failed_submission_removes_pending_context(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)

        async def fail(msg: Any) -> None:
            raise RuntimeError("Synthetic transport failure")

        monkeypatch.setattr(state.send_tokenizer, "put", fail)
        with pytest.raises(RuntimeError, match="Synthetic transport failure"):
            await api_server.tts_generate(
                api_server.TTSGenerateRequest(text="First.", retain_continuation=True), "owner",
            )
        assert not state.ack_map and not state.event_map and not state.start_time_map
        assert not state.continuations._entries

    asyncio.run(scenario())


def test_worker_cancellation_drops_late_frames_and_keeps_other_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _WorkerHarness(monkeypatch)
    sent = []
    reset = []
    vocoder_reset = vocoder.TTSVocoderManager.reset

    def observe_reset(manager: vocoder.TTSVocoderManager, uid: int) -> None:
        reset.append(uid)
        vocoder_reset(manager, uid)

    def submit(msg: Any) -> None:
        sent.append(msg)
        if isinstance(msg, TTSUserMsg):
            worker._backend(msg)

    monkeypatch.setattr(vocoder.TTSVocoderManager, "reset", observe_reset)
    monkeypatch.setattr(
        tokenizer_server, "ZmqPushQueue", lambda addr, **kwargs: SimpleNamespace(
            put=submit if addr == "backend" else worker._frontend
        ),
    )
    # Cancel an already submitted request, before its queued frames arrive.
    abort = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(AbortMsg(3)))
    worker.pending.extend([replace(_template(3), retain_continuation=True), abort])
    replies = worker.run(_template(4))
    assert TTSCancelMsg(3) in sent
    assert {r.uid for r in replies} == {4}
    assert 3 in reset


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("stall", [False, True])
def test_http_worker_failure_is_terminal(monkeypatch: pytest.MonkeyPatch, stream: bool, stall: bool) -> None:
    worker = _WorkerHarness(monkeypatch)
    incoming: asyncio.Queue[TTSAudioReply] = asyncio.Queue()
    state = _frontend(monkeypatch, worker, incoming)
    state.worker_timeout = 0.03
    cancelled = []

    async def submit(msg: TTSTokenizeMsg | AbortMsg) -> None:
        if isinstance(msg, AbortMsg):
            cancelled.append(msg.uid)
        elif not stall:
            assert isinstance(msg, TTSTokenizeMsg)
            for reply in worker.run(replace(msg, max_prompt_frames=0)):
                await incoming.put(reply)

    monkeypatch.setattr(state.send_tokenizer, "put", submit)
    headers = {"X-TTS-Session-ID": "owner"}
    with TestClient(api_server.app) as client:
        response = client.post("/tts/generate", headers=headers, json={
            "text": "First.", "stream": stream, "retain_continuation": True,
        })
        assert response.status_code == (504 if stall else 400)
        assert response.headers["X-TTS-Generation-Status"] == "failed"
        if stream:
            status = client.get(response.headers["X-TTS-Continuation-Status-URL"], headers=headers)
            assert status.json()["finish_reason"] == "failed"
            assert not status.json()["usable"]
    assert not state.active_workers and not state.ack_map
    assert cancelled == ([0] if stall else [])


@pytest.mark.parametrize("stall", [False, True])
def test_retained_stream_failure_after_headers(monkeypatch: pytest.MonkeyPatch, stall: bool) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        incoming: asyncio.Queue[TTSAudioReply] = asyncio.Queue()
        state = _frontend(monkeypatch, worker, incoming)
        state.worker_timeout = 0.03
        sent: asyncio.Queue[TTSTokenizeMsg | AbortMsg] = asyncio.Queue()
        monkeypatch.setattr(state.send_tokenizer, "put", sent.put)
        response = await api_server.tts_generate(
            api_server.TTSGenerateRequest(text="First.", stream=True, retain_continuation=True),
            "owner",
        )
        msg = sent.get_nowait()
        assert isinstance(msg, TTSTokenizeMsg)
        await incoming.put(TTSAudioReply(msg.uid, b"\0" * 4, False))
        received = []

        async def send(event: Message) -> None:
            received.append(event)
            if event.get("more_body") and not stall:
                await incoming.put(TTSAudioReply(
                    msg.uid, b"", True, error="oversized next chunk", finish_reason="failed",
                ))

        async def receive() -> Message:
            return await asyncio.Future[Message]()

        await asyncio.wait_for(response({"type": "http"}, receive, send), timeout=2)
        assert received[0]["status"] == 200
        assert received[1]["body"] == b"\0" * 4
        assert received[-1]["more_body"] is False
        status = state.continuations.status(response.headers["X-TTS-Continuation-ID"], "owner")
        assert not status.usable and status.finish_reason == "failed"
        if stall:
            assert await asyncio.wait_for(sent.get(), timeout=2) == AbortMsg(msg.uid)
        else:
            assert sent.empty()
        assert not state.active_workers and not state.ack_map

    asyncio.run(scenario())


def test_buffered_http_disconnect_cancels_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _WorkerHarness(monkeypatch)

    async def scenario() -> None:
        state = _frontend(monkeypatch, worker)
        sent: asyncio.Queue[TTSTokenizeMsg | AbortMsg] = asyncio.Queue()
        monkeypatch.setattr(state.send_tokenizer, "put", sent.put)
        submitted = asyncio.Event()

        async def receive() -> Message:
            await submitted.wait()
            return {"type": "http.disconnect"}

        request = api_server.Request({"type": "http"}, receive=receive)
        pending = asyncio.create_task(api_server._tts_generate_http(
            api_server.TTSGenerateRequest(text="First.", stream=False, retain_continuation=True),
            request,
            "owner",
        ))
        msg = await asyncio.wait_for(sent.get(), timeout=2)
        assert isinstance(msg, TTSTokenizeMsg)
        submitted.set()
        assert (await asyncio.wait_for(pending, timeout=2)).status_code == 499
        assert await asyncio.wait_for(sent.get(), timeout=2) == AbortMsg(msg.uid)
        assert not state.ack_map and not state.active_workers and not state.continuations._entries

    asyncio.run(scenario())
