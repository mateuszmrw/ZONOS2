"""Render blank-line-separated text to a WAV through the ZONOS2 HTTP API."""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import tempfile
import time
import uuid
import wave
from array import array
from dataclasses import dataclass
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class ChapterError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class _NoRedirects(HTTPRedirectHandler):
    # Do not send a session header or speaker upload to a redirect target.
    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str,
    ) -> None:
        return None


class TTSClient:
    def __init__(self, base_url: str, session_id: str, timeout: float = 90.0) -> None:
        if urlsplit(base_url).scheme not in ("http", "https") or not urlsplit(base_url).hostname:
            raise ValueError("Use an HTTP or HTTPS base URL.")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Timeout must be positive and finite.")
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self.timeout = timeout
        self.opener = build_opener(_NoRedirects)

    def _open(self, path: str, body: dict[str, Any] | None = None) -> HTTPResponse:
        response = self.opener.open(
            Request(
                self.base_url + path,
                data=None if body is None else json.dumps(body).encode("utf-8"),
                headers={"X-TTS-Session-ID": self.session_id, "Content-Type": "application/json"},
            ),
            timeout=self.timeout,
        )
        return cast(HTTPResponse, response)

    @staticmethod
    def _http_error(exc: HTTPError) -> ChapterError:
        with exc:
            try:
                detail = json.loads(exc.read(4096)).get("detail", exc.reason)
            except (ValueError, OSError, AttributeError, HTTPException):
                detail = exc.reason
        return ChapterError(
            f"HTTP {exc.code}: {detail}",
            retryable=exc.code in {429, 500, 502, 503, 504},
        )

    def upload_speaker(self, path: Path) -> str:
        body = {
            "speaker_audio_name": path.name,
            "speaker_audio_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
        }
        try:
            with self._open("/tts/speakers", body) as response:
                speaker_id = json.load(response)["id"]
                if not isinstance(speaker_id, str) or not speaker_id:
                    raise ValueError("missing speaker ID")
                return speaker_id
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except (ValueError, KeyError, TypeError, OSError, HTTPException) as exc:
            raise ChapterError(f"Could not upload speaker: {exc}") from exc

    def generate(self, body: dict[str, Any]) -> tuple[bytes, str]:
        """Read one retained response. Return audio only after its ID is usable."""
        try:
            with self._open("/tts/generate", body) as response:
                headers = response.headers
                if (
                    headers.get_content_type() != "audio/pcm"
                    or headers.get("X-Audio-Sample-Rate") != "44100"
                    or headers.get("X-Audio-Channels") != "1"
                    or headers.get("X-Audio-Format") != "float32"
                ):
                    raise ChapterError("Unexpected PCM format.")
                audio = bytearray()
                while block := response.read(65536):
                    audio.extend(block)
                length = headers.get("Content-Length")
                if length is not None and int(length) != len(audio):
                    raise ChapterError("The HTTP audio body was cut short.", retryable=True)
            child = headers.get("X-TTS-Continuation-ID")
            status_path = headers.get("X-TTS-Continuation-Status-URL")
            if (
                not child
                or not re.fullmatch("[0-9a-f]{32}", child)
                or status_path != f"/tts/continuations/{child}"
            ):
                raise ChapterError("Missing or invalid continuation status URL.")
            # The server sets the terminal status before it closes the PCM body.
            # Treat a nonterminal result as a failed attempt, not as accepted audio.
            with self._open(status_path) as response:
                status = json.load(response)
            reason = status["finish_reason"]
            if reason != "complete" or status["usable"] is not True:
                raise ChapterError(
                    f"{reason or status['status']}: {status.get('error') or 'context not ready'}",
                    retryable=reason in ("frame_limit", "failed", None),
                )
            if not body["stream"] and headers.get("X-TTS-Generation-Status") != "complete":
                raise ChapterError("Buffered generation status disagrees with the status URL.")
            return bytes(audio), child
        except HTTPError as exc:
            raise self._http_error(exc) from exc
        except (URLError, OSError, HTTPException) as exc:
            raise ChapterError(f"HTTP connection failed: {exc}", retryable=True) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise ChapterError("Invalid HTTP response from ZONOS2.") from exc


def _pcm16(pcm: bytes) -> bytes:
    if not pcm or len(pcm) % 4:
        raise ChapterError("Empty or unaligned PCM response.", retryable=True)
    floats = array("f")
    floats.frombytes(pcm)
    if sys.byteorder == "big":
        floats.byteswap()
    if not all(map(math.isfinite, floats)):
        raise ChapterError("Non-finite PCM sample.", retryable=True)
    samples = array("h", (round(max(-1.0, min(1.0, value)) * 32767) for value in floats))
    if sys.byteorder == "big":
        samples.byteswap()
    return samples.tobytes()


@dataclass(frozen=True)
class ChapterChunk:
    samples: int
    attempts: int
    continuation_id: str


def render_chapter(
    client: TTSClient,
    text: str,
    output: Path,
    *,
    stream: bool = True,
    speaker_id: str | None = None,
    language: str = "en_us",
    max_tokens: int = 1200,
    retry_max_tokens: int | None = None,
    retries: int = 2,
    seed: int = 123,
) -> list[ChapterChunk]:
    paragraphs = [p.strip() for p in re.split(r"\n[^\S\n]*\n", text.strip()) if p.strip()]
    if not paragraphs:
        raise ValueError("The chapter has no text.")
    if max_tokens < 1 or retries < 0 or (retry_max_tokens is not None and retry_max_tokens < max_tokens):
        raise ValueError("Use a positive frame budget, a retry budget at least as large, and retries >= 0.")
    parent = None
    chunks: list[ChapterChunk] = []
    tmp_path = None
    try:
        # Keep the destination unchanged if any paragraph fails after retries.
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".chapter-", delete=False) as temp:
            tmp_path = Path(temp.name)
            with wave.open(temp, "wb") as wav:
                wav.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
                for index, paragraph in enumerate(paragraphs, start=1):
                    body: dict[str, Any] = {
                        "text": paragraph, "stream": stream, "retain_continuation": True,
                    }
                    if parent is None:
                        body.update(language=language, long_form=True, long_form_split_mode="sentence")
                        if speaker_id is not None:
                            body["speaker_embedding_id"] = speaker_id
                    else:
                        body["continuation_of"] = parent
                    for attempt in range(retries + 1):
                        try:
                            pcm, child = client.generate({
                                **body,
                                "max_tokens": max_tokens if not attempt else retry_max_tokens or 2 * max_tokens,
                                "seed": seed + attempt,
                            })
                            accepted = _pcm16(pcm)
                        except ChapterError as exc:
                            if attempt == retries or not exc.retryable:
                                where = f" Last usable ID: {parent}." if parent else ""
                                raise ChapterError(f"Paragraph {index} failed: {exc}.{where}") from exc
                            time.sleep(min(2**attempt * 0.25, 2.0))
                        else:
                            wav.writeframesraw(accepted)
                            parent = child
                            chunks.append(ChapterChunk(len(accepted) // 2, attempt + 1, child))
                            break
        os.replace(tmp_path, output)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chapter", type=Path, help="UTF-8 text file; put a blank line between paragraphs")
    parser.add_argument("output", type=Path, help="output WAV (replaced only after success)")
    parser.add_argument("--base-url", default="http://127.0.0.1:1919")
    parser.add_argument("--session-id")
    speaker = parser.add_mutually_exclusive_group()
    speaker.add_argument("--speaker", type=Path, help="upload a speaker reference in this session")
    speaker.add_argument("--speaker-id", help="use a speaker ID from --session-id")
    parser.add_argument("--language", default="en_us")
    parser.add_argument("--buffered", action="store_true", help="use buffered instead of streaming HTTP")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP socket timeout in seconds")
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--retry-max-tokens", type=int, help="frame budget on retry (default: twice --max-tokens)")
    parser.add_argument("--retries", type=int, default=2, help="retries per paragraph")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.speaker_id and not args.session_id:
        parser.error("--speaker-id requires --session-id")
    if args.chapter.resolve() == args.output.resolve():
        parser.error("Use different input and output paths.")
    try:
        text = args.chapter.read_text(encoding="utf-8")
        client = TTSClient(args.base_url, args.session_id or uuid.uuid4().hex, args.timeout)
        speaker_id = client.upload_speaker(args.speaker) if args.speaker else args.speaker_id
        chunks = render_chapter(
            client, text, args.output, stream=not args.buffered, speaker_id=speaker_id,
            language=args.language, max_tokens=args.max_tokens, retry_max_tokens=args.retry_max_tokens,
            retries=args.retries, seed=args.seed,
        )
    except (ChapterError, OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Wrote {args.output}: {len(chunks)} paragraphs, {sum(c.attempts - 1 for c in chunks)} retries.")


if __name__ == "__main__":
    main()
