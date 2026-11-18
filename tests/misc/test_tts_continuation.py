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
from zonos2.message.tts import TTSContinuationContext, TTSFinishReason, TTSUserMsg
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
    decoded: object = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(cast(Any, msg)))
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
@pytest.mark.parametrize("early_eos", [None, 30])
def test_http_reports_truncation_and_retries_from_last_good_id(
    monkeypatch: pytest.MonkeyPatch, stream: bool, early_eos: int | None
) -> None:
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
        worker.eos_frame = early_eos
        worker.finish_reason = "frame_limit"
        failed = client.post(
            "/tts/generate",
            headers=headers,
            json={"text": "This request cannot finish.", "continuation_of": parent, "stream": stream},
        )
        assert failed.status_code == 200
        if not stream:
            assert failed.headers["X-TTS-Generation-Status"] == "frame_limit"
        status_url = failed.headers["X-TTS-Continuation-Status-URL"]
        status = client.get(status_url, headers=headers)
        assert status.headers["Cache-Control"] == "no-store"
        assert status.json()["finish_reason"] == "frame_limit"
        assert status.json()["usable"] is False
        assert client.get(status_url, headers={"X-TTS-Session-ID": "other"}).status_code == 404
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
        np.testing.assert_allclose(pcm, 103.0, rtol=0, atol=2e-5)


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
        submitted: asyncio.Queue[TTSTokenizeMsg] = asyncio.Queue()
        waiting = asyncio.Event()
        disconnected = asyncio.Event()
        wait_for_tts_start = state.wait_for_tts_start

        async def observe_start(uid: int) -> None:
            waiting.set()
            await wait_for_tts_start(uid)

        with monkeypatch.context() as mp:
            mp.setattr(state, "send_one", submitted.put)
            mp.setattr(state, "wait_for_tts_start", observe_start)
            response_task = asyncio.create_task(api_server.tts_generate(
                api_server.TTSGenerateRequest(text="Next.", continuation_of=parent_id, stream=True),
                "owner",
            ))
            msg = await asyncio.wait_for(submitted.get(), timeout=2)
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
            if spec == "2.4":
                with pytest.raises(ClientDisconnect):
                    await asyncio.wait_for(exchange, timeout=2)
            else:
                await asyncio.wait_for(exchange, timeout=2)

        assert not state.ack_map and not state.event_map and not state.start_time_map
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
