"""Checkpoint sharding and CUDA HTTP tests. Set ZONOS2_TEST_MODEL to run inference."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator

import httpx
import pytest
import torch
import uvicorn
from starlette.types import Receive, Scope, Send

from scripts.tts_chapter import TTSClient, render_chapter

MODEL = os.getenv("ZONOS2_TEST_MODEL")
ROOT = Path(__file__).resolve().parents[2]
VOICE = ROOT / "default_voices" / "AmericanFemale.mp3"
CHAPTER = """
The first light fell across the quiet harbor. Mara opened the door and stepped outside.

A small boat rocked beside the pier. Someone had left a red coat on the rail.

She picked it up and heard a bell across the water. The ferry was already on its way.

When the passengers stepped ashore, she watched their faces. A boy waved to her from the back.

He held a letter from her sister. Mara read it once, and then she read it again.

The city could wait until tomorrow. She folded the letter and walked home.
"""


def _port() -> int:
    # Reserve room for the API and its port + 1 rendezvous (released before launch).
    for _ in range(20):
        with socket.socket() as api, socket.socket() as ranks:
            api.bind(("127.0.0.1", 0))
            port = int(api.getsockname()[1])
            try:
                ranks.bind(("127.0.0.1", port + 1))
            except (OSError, OverflowError):
                continue
            return port
    raise RuntimeError("No adjacent ports available for the API and its workers")


@contextmanager
def _serve(devices: list[str], tp: int, log_path: Path) -> Iterator[str]:
    port = _port()
    url = f"http://127.0.0.1:{port}"
    assert MODEL is not None
    args = [
        sys.executable, "-u", "-m", "zonos2", "--model-path", MODEL,
        "--host", "127.0.0.1", "--port", str(port), "--num-tokenizer", "0",
        "--tp-size", str(tp), "--max-running-requests", "2", "--cuda-graph-max-bs", "2",
        "--memory-ratio", "0.75", "--attention-backend", "fi",
    ]
    if tp > 1:
        # Use PyTorch's NCCL runtime. The optional PyNCCL extension also needs
        # system NCCL development libraries (not included in many GPU images).
        args.append("--disable-pynccl")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": ",".join(devices),
        "PYTHONPATH": str(ROOT / "python"),
        "ZONOS2_CONTINUATION_TTL_SECONDS": "900",
        "ZONOS2_TTS_WORKER_TIMEOUT_SECONDS": "90",
    }
    with log_path.open("w") as log:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=log, env=env, start_new_session=True)
        try:
            with httpx.Client(timeout=2, trust_env=False) as http:
                deadline = time.monotonic() + 360
                while proc.poll() is None and time.monotonic() < deadline:
                    startup_log = log_path.read_text()
                    if "Traceback (most recent call last)" in startup_log:
                        pytest.fail(f"Server worker failed to start: {log_path}\n{startup_log[-5000:]}")
                    try:
                        response = http.get(url + "/tts/capabilities")
                        if response.is_success:
                            assert response.json()["speaker_enabled"], "this test needs a speaker checkpoint"
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.5)
                else:
                    pytest.fail(f"Server failed to start: {log_path}\n{log_path.read_text()[-5000:]}")
            yield url
        finally:
            # A frontend exit does not necessarily stop its spawned GPU workers.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


def _replica(session: str) -> int:
    return hashlib.sha256(session.encode()).digest()[0] % 2


@contextmanager
def _sticky_proxy(upstream: list[str], routes: list[tuple[str, str, int]]) -> Iterator[str]:
    """Loopback test router; forward PCM as it arrives, without buffering or failover."""
    async def proxy(scope: Scope, receive: Receive, send: Send) -> None:
        headers = dict(scope["headers"])
        session = headers.get(b"x-tts-session-id", b"").decode("ascii")
        index = _replica(session)
        path = scope["path"]
        routes.append((session, path, index))
        body = b""
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body += event.get("body", b"")
            if not event.get("more_body"):
                break

        async def forward() -> None:
            async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
                async with client.stream(
                    scope["method"], upstream[index] + path, content=body,
                    headers={
                        "X-TTS-Session-ID": session,
                        "Content-Type": headers.get(b"content-type", b"application/json").decode(),
                    },
                ) as response:
                    out_headers = [
                        (k, v) for k, v in response.headers.raw if k.lower() != b"transfer-encoding"
                    ]
                    await send({"type": "http.response.start", "status": response.status_code,
                                "headers": out_headers})
                    async for chunk in response.aiter_raw():
                        await send({"type": "http.response.body", "body": chunk, "more_body": True})
                    await send({"type": "http.response.body", "body": b"", "more_body": False})

        forwarding = asyncio.create_task(forward())
        disconnected = asyncio.ensure_future(receive())
        try:
            await asyncio.wait((forwarding, disconnected), return_when=asyncio.FIRST_COMPLETED)
            if forwarding.done():
                forwarding.result()
        finally:
            for task in (forwarding, disconnected):
                task.cancel()
            await asyncio.gather(forwarding, disconnected, return_exceptions=True)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        config = uvicorn.Config(proxy, interface="asgi3", lifespan="off", log_level="warning")
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        thread.start()
        try:
            deadline = time.monotonic() + 5
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert server.started
            yield url
        finally:
            server.should_exit = True
            thread.join(timeout=5)


class _TruncateSecondParagraph(TTSClient):
    """Force one real frame limit inside the normal chapter-client retry loop."""

    failed_parent: str | None = None

    def generate(self, body: dict) -> tuple[bytes, str]:
        if self.failed_parent is None and "continuation_of" in body:
            self.failed_parent = body["continuation_of"]
            body = {**body, "max_tokens": 16}
        return super().generate(body)


async def _disconnect_during_overlap(url: str, session: str, parent: str) -> dict:
    headers = {"X-TTS-Session-ID": session}
    body = {"continuation_of": parent, "retain_continuation": True, "stream": True, "seed": None}
    async with httpx.AsyncClient(base_url=url, headers=headers, timeout=90, trust_env=False) as http:
        sibling_playing = asyncio.Event()

        async def abandon() -> tuple[str, int]:
            message = "At dawn, the lighthouse went dark and the boats began to stir. "
            async with http.stream("POST", "/tts/generate", json={
                **body, "text": message * 14, "topk": 100, "min_p": 0.16,
            }) as response:
                response.raise_for_status()
                child_path = response.headers["X-TTS-Continuation-Status-URL"]
                # Keep the iterator alive until the response context closes.
                # Finalizing a temporary iterator can close the HTTP stream.
                playback = response.aiter_raw()
                audio = await anext(playback)
                assert audio
                await asyncio.wait_for(sibling_playing.wait(), timeout=15)
                status = (await http.get(child_path)).raise_for_status().json()
                assert status["status"] == "pending"
            return child_path, len(audio)

        async def finish() -> tuple[str, int]:
            async with http.stream("POST", "/tts/generate", json={
                **body, "text": "The other boat sailed south, past the end of the long stone pier.",
                "topk": 90, "top_p": 0.9, "min_p": 0.2,
            }) as response:
                response.raise_for_status()
                child_path = response.headers["X-TTS-Continuation-Status-URL"]
                playback = response.aiter_raw()
                first = await anext(playback)
                assert first
                sibling_playing.set()
                remaining = [block async for block in playback]
                audio = first + b"".join(remaining)
            status = (await http.get(child_path)).raise_for_status().json()
            assert audio and status["usable"] and status["finish_reason"] == "complete"
            return child_path, len(audio)

        (abandoned, first_bytes), (sibling, sibling_bytes) = await asyncio.gather(abandon(), finish())
        assert abandoned != sibling
        deadline = time.monotonic() + 10
        while (await http.get(abandoned)).status_code != 404:
            assert time.monotonic() < deadline, "abandoned continuation was not removed"
            await asyncio.sleep(0.05)
        return {
            "overlapped_audio": True,
            "abandoned_status": 404,
            "abandoned_bytes": first_bytes,
            "sibling_bytes": sibling_bytes,
        }


def _exercise(url: str, session: str, output: Path, scheduler_log: Path) -> tuple[str, str]:
    client = TTSClient(url, session)
    speaker = client.upload_speaker(VOICE)
    records: dict = {}
    for stream in (False, True):
        name = "streamed" if stream else "buffered"
        wav_path = output / f"{name}.wav"
        runner = _TruncateSecondParagraph(url, session)
        chunks = render_chapter(runner, CHAPTER, wav_path, speaker_id=speaker, stream=stream)
        assert len(chunks) == 6 and all(chunk.samples > 0 for chunk in chunks)
        assert chunks[1].attempts >= 2
        assert runner.failed_parent == chunks[0].continuation_id
        with wave.open(str(wav_path)) as wav:
            assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 44100)
            assert wav.getnframes() == sum(c.samples for c in chunks)
        records[name] = [{"samples": c.samples, "attempts": c.attempts} for c in chunks]
    parent = chunks[0].continuation_id
    headers = {"X-TTS-Session-ID": session}
    with httpx.Client(base_url=url, headers=headers, timeout=90, trust_env=False) as http:
        parent_status = f"/tts/continuations/{parent}"
        cutoff = http.post("/tts/generate", json={
            "continuation_of": parent, "text": "This branch cannot reach the end.",
            "max_tokens": 16, "stream": False,
        }).raise_for_status()
        assert cutoff.headers["X-TTS-Generation-Status"] == "frame_limit"
        assert len(cutoff.content) == (16 - 8) * 512 * 4
        bad = http.get(cutoff.headers["X-TTS-Continuation-Status-URL"]).raise_for_status().json()
        assert bad["finish_reason"] == "frame_limit" and not bad["usable"]
        assert http.post("/tts/generate", json={
            "text": "Cannot use incomplete context.",
            "continuation_of": cutoff.headers["X-TTS-Continuation-ID"],
        }).status_code == 409
        assert http.get(parent_status).raise_for_status().json()["usable"]
        records["truncation"] = {"finish_reason": bad["finish_reason"], "samples": len(cutoff.content) // 4}

    records["disconnect"] = asyncio.run(_disconnect_during_overlap(url, session, parent))
    assert "cancelled" in scheduler_log.read_text(), "scheduler did not acknowledge any cancellation"
    retry_audio, child = client.generate({
        "continuation_of": parent, "text": "The boats returned before the rain.",
        "seed": 123, "stream": False, "retain_continuation": True,
    })
    assert retry_audio and child != parent
    records["disconnect"]["retry_samples"] = len(retry_audio) // 4
    (output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    return parent, speaker


@pytest.mark.parametrize("rank", [0, 1])
def test_tp_checkpoint_keeps_full_audio_output_head(
    rank: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zonos2.distributed import DistributedInfo
    from zonos2.models import weight

    monkeypatch.setattr(weight, "get_tp_info", lambda: DistributedInfo(rank, 2))
    head = torch.arange(24).reshape(6, 4)
    embedding = torch.arange(32).reshape(8, 4)
    state = {
        "multi_output.weight": head,
        "multi_embedder.embedders.0.weight": embedding,
    }
    shard = weight._shard_state_dict(state)
    assert torch.equal(shard["multi_output.weight"], head)
    assert torch.equal(shard["multi_embedder.embedders.0.weight"], embedding.chunk(2)[rank])


@pytest.mark.gpu
@pytest.mark.skipif(not MODEL, reason="set ZONOS2_TEST_MODEL to run (loads weights)")
@pytest.mark.parametrize("topology", ["single", "sticky", "tp2"])
def test_http_continuation_on_cuda(topology: str, tmp_path: Path) -> None:
    needed = 1 if topology == "single" else 2
    if torch.cuda.device_count() < needed:
        pytest.skip(f"{topology} needs {needed} visible CUDA devices")
    device_ids = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
    if device_ids == [""]:
        device_ids = [str(i) for i in range(torch.cuda.device_count())]
    output = Path(os.getenv("ZONOS2_TEST_OUTPUT", str(tmp_path))) / topology
    output.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        log = output / "server.log"
        if topology == "tp2":
            url = stack.enter_context(_serve(device_ids[:2], 2, log))
        else:
            url = stack.enter_context(_serve(device_ids[:1], 1, log))
        if topology != "sticky":
            _exercise(url, f"chapter-{topology}", output, log)
            return

        logs = [log, output / "replica-b.log"]
        backends = [url, stack.enter_context(_serve(device_ids[1:2], 1, logs[1]))]
        routes: list[tuple[str, str, int]] = []
        proxy = stack.enter_context(_sticky_proxy(backends, routes))
        candidates = [f"chapter-{n}" for n in range(100)]
        sessions = [next(s for s in candidates if _replica(s) == i) for i in (0, 1)]
        session, second_session = sessions
        # Use both replicas, and first test each failure against a known-good ID.
        other = TTSClient(proxy, second_session)
        other_speaker = other.upload_speaker(VOICE)
        _, other_id = other.generate({
            "text": "This second session has its own home.", "speaker_embedding_id": other_speaker,
            "stream": False, "retain_continuation": True, "seed": 123,
        })
        followup, other_child = other.generate({
            "text": "It stays there for the rest of the chain.", "continuation_of": other_id,
            "stream": True, "retain_continuation": True, "seed": 123,
        })
        assert followup and other_id != other_child
        parent, speaker = _exercise(proxy, session, output, logs[0])
        with httpx.Client(timeout=10, trust_env=False) as http:
            for key, owner, home in ((parent, session, 0), (other_id, second_session, 1)):
                headers = {"X-TTS-Session-ID": owner}
                path = f"/tts/continuations/{key}"
                assert http.get(backends[home] + path, headers=headers).raise_for_status().json()["usable"]
                wrong = backends[1 - home]
                assert http.get(wrong + path, headers=headers).status_code == 404
                assert http.post(wrong + "/tts/generate", headers=headers, json={
                    "text": "Wrong replica.", "continuation_of": key,
                }).status_code == 404
            headers = {"X-TTS-Session-ID": session}
            preview = f"/tts/speakers/{speaker}/preview"
            assert http.get(backends[0] + preview, headers=headers).status_code == 200
            unknown_speaker = http.get(backends[1] + preview, headers=headers)
            assert unknown_speaker.status_code == 400
            assert unknown_speaker.json()["detail"] == f"Unknown cached speaker id '{speaker}'."
        assert {i for _, _, i in routes} == {0, 1}
        for owner, index in zip(sessions, (0, 1), strict=True):
            paths = {path for routed, path, i in routes if routed == owner and i == index}
            assert {"/tts/speakers", "/tts/generate"} <= paths
            assert any(path.startswith("/tts/continuations/") for path in paths)
        (output / "routing.json").write_text(json.dumps({
            "speaker_upload_generation_and_status_used_sticky_routing": True,
            "sessions_chained": len(sessions),
            "wrong_replica_status": 404,
            "wrong_replica_generation": 404,
            "wrong_replica_speaker_preview": unknown_speaker.status_code,
        }, indent=2) + "\n")
