"""Exercise the chapter client against a real loopback HTTP connection."""

import json
import struct
import threading
import wave
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from scripts.tts_chapter import ChapterError, TTSClient, render_chapter


@contextmanager
def _server(disconnect: bool = False) -> Iterator[tuple[TTSClient, list[dict]]]:
    requests: list[dict] = []
    statuses: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.headers["X-TTS-Session-ID"] == "chapter-test"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            index = len(requests)
            key = f"{index:032x}"
            status = "frame_limit" if body["max_tokens"] == 16 and index > 1 else "complete"
            statuses[key] = status
            pcm = struct.pack("<ff", index / 8, -index / 8)
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("Content-Length", str(len(pcm)))
            self.send_header("X-Audio-Sample-Rate", "44100")
            self.send_header("X-Audio-Channels", "1")
            self.send_header("X-Audio-Format", "float32")
            self.send_header("X-TTS-Continuation-ID", key)
            self.send_header("X-TTS-Continuation-Status-URL", f"/tts/continuations/{key}")
            if not body["stream"]:
                self.send_header("X-TTS-Generation-Status", status)
            self.end_headers()
            self.wfile.write(pcm[:4] if disconnect and index == 2 else pcm)

        def do_GET(self) -> None:
            assert self.headers["X-TTS-Session-ID"] == "chapter-test"
            status = statuses[self.path.split("/")[-1]]
            result = json.dumps({
                "usable": status == "complete", "finish_reason": status, "status": status,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(result)))
            self.end_headers()
            self.wfile.write(result)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield TTSClient(f"http://127.0.0.1:{server.server_port}", "chapter-test", 3), requests
        finally:
            server.shutdown()
            thread.join(timeout=3)


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize("disconnect", [True, False])
def test_chapter_retries_same_parent_and_writes_only_accepted_pcm(
    tmp_path: Path, stream: bool, disconnect: bool,
) -> None:
    output = tmp_path / "chapter.wav"
    with _server(disconnect) as (client, requests):
        chunks = render_chapter(
            client, "First.\n\nSecond.\n\nLast.", output,
            stream=stream, max_tokens=16, retry_max_tokens=64,
        )
        assert [chunk.attempts for chunk in chunks] == [1, 2, 2]
        assert [body.get("continuation_of") for body in requests] == [
            None, chunks[0].continuation_id, chunks[0].continuation_id,
            chunks[1].continuation_id, chunks[1].continuation_id,
        ]
        assert all(body["stream"] is stream for body in requests)
        assert "language" in requests[0] and "language" not in requests[1]
    with wave.open(str(output)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) == (44100, 1, 2, 6)
        wanted = [round(value / 8 * 32767) for i in (1, 3, 5) for value in (i, -i)]
        assert struct.unpack("<6h", wav.readframes(6)) == tuple(wanted)


def test_chapter_failure_does_not_replace_destination(tmp_path: Path) -> None:
    output = tmp_path / "existing.wav"
    output.write_bytes(b"keep this")
    with _server() as (client, _):
        with pytest.raises(ChapterError, match="Paragraph 2 failed: frame_limit"):
            render_chapter(client, "First.\n\nSecond.", output, max_tokens=16, retries=0)
    assert output.read_bytes() == b"keep this"
    assert list(tmp_path.iterdir()) == [output]
