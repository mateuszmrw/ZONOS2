from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.responses import Response
from fastapi.testclient import TestClient
from zonos2.message.tts import TTSTokenizeMsg
from zonos2.server import api_server
from zonos2.server.fixed_references import (
    FixedReference,
    load_fixed_references,
    select_fixed_reference,
)

REVISION = "a" * 40
LANGUAGE = "fr_fr"
SEED = 42


def _asset(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    voices = tmp_path / "voices"
    voices.mkdir()
    voice = voices / "Example.npy"
    np.save(voice, np.array([1, 2, 3], dtype=np.float32))
    params = b'{"n_codebooks":9}'
    (tmp_path / "params.json").write_bytes(params)
    return tmp_path / "references.json", voice, {
        "version": 1,
        "model_revision": REVISION,
        "params_sha256": hashlib.sha256(params).hexdigest(),
        "references": [{
            "speaker_file": voice.name,
            "speaker_sha256": hashlib.sha256(voice.read_bytes()).hexdigest(),
            "language": LANGUAGE,
            "seed": SEED,
            "text": "Une voix.",
            "speech_frames": 1,
            "prompt_codes": [[42] * 9],
        }],
    }


@pytest.mark.parametrize(
    "invalid",
    [None, "model", "params", "language", "duplicate", "ambiguous", "speaker", "width", "limit"],
)
def test_fixed_reference_validation(tmp_path: Path, invalid: str | None) -> None:
    path, voice, data = _asset(tmp_path)
    first = data["references"][0]
    replacements: dict[str, dict[str, object]] = {
        "language": {"language": "not_supported"},
        "speaker": {"speaker_sha256": "0" * 64},
        "width": {"prompt_codes": [[42] * 8]},
        "limit": {"speech_frames": 2},
    }
    if invalid is not None and invalid in replacements:
        data = {**data, "references": [{**first, **replacements[invalid]}]}
    elif invalid == "duplicate":
        data = {**data, "references": [first, first]}
    elif invalid == "model":
        data = {**data, "model_revision": "b" * 40}
    elif invalid == "params":
        data = {**data, "params_sha256": "0" * 64}
    path.write_text(json.dumps(data), encoding="utf-8")
    candidates = [("default_example", voice)]
    if invalid == "ambiguous":
        candidates = [*candidates, ("default_same_name", voice)]

    def load() -> Mapping[tuple[str, str], FixedReference]:
        return load_fixed_references(
            path, model_dir=tmp_path, model_revision=REVISION, n_codebooks=9,
            max_seq_len=2048, languages={LANGUAGE}, voices=candidates,
        )

    if invalid is not None:
        with pytest.raises(ValueError):
            load()
        return

    loaded = load()
    reference = loaded[LANGUAGE, "default_example"]
    assert reference.prompt_codes == ((42,) * 9,)
    for lang, speaker, seed, expected in (
        (LANGUAGE, "default_example", SEED, reference),
        (LANGUAGE, "default_example", 123, None),
        (LANGUAGE, "default_example", None, None),
        (LANGUAGE, "default_missing", SEED, None),
        ("en_gb", "default_example", SEED, None),
    ):
        assert select_fixed_reference(loaded, language=lang, speaker_id=speaker, seed=seed) is expected


def _frontend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, num_tokenizer: int = 0,
) -> api_server.FrontendManager:
    config: Any = SimpleNamespace(
        model_path=tmp_path, tts_default_voices_dir=str(tmp_path / "voices"),
        num_tokenizer=num_tokenizer, max_seq_len=2048, tts_n_codebooks=9,
        model_config=SimpleNamespace(speaker_enabled=True, speaker_embedding_dim=3),
    )
    queue: Any = SimpleNamespace(stop=lambda: None)
    state = api_server.FrontendManager(config, queue, queue)
    monkeypatch.setattr(api_server, "_GLOBAL_STATE", state)
    return state


def test_fixed_references_are_loaded_before_http_and_do_not_change_between_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, voice, data = _asset(tmp_path)
    content = json.dumps(data)
    path.write_text(content, encoding="utf-8")
    monkeypatch.setenv("ZONOS2_FIXED_REFERENCE_FILE", str(path))
    monkeypatch.setenv("ZONOS2_MODEL_REVISION", REVISION)
    state = _frontend(tmp_path, monkeypatch)
    speaker_id = api_server._default_voice_id(voice.parent, voice)
    submitted: list[TTSTokenizeMsg] = []

    async def submit(
        _: api_server.FrontendManager,
        msg: TTSTokenizeMsg,
        req: api_server.TTSGenerateRequest,
        session_id: str | None,
    ) -> Response:
        submitted.append(msg)
        return Response(b"test audio")

    monkeypatch.setattr(api_server, "_submit_tts_generation", submit)
    with TestClient(api_server.app) as client:
        caps = client.get("/tts/capabilities").raise_for_status().json()
        assert caps["fixed_references"] == {LANGUAGE: {speaker_id: SEED}}
        original = state.fixed_references[LANGUAGE, speaker_id]
        body = {"text": "Bonjour.", "language": LANGUAGE, "speaker_embedding_id": speaker_id}
        for options in (
            {},
            {"seed": SEED},
            {"seed": 123},
            {"seed": SEED, "retain_continuation": True},
            {"seed": SEED, "defer_audio_tail": True},
            {"seed": SEED},
        ):
            reply = client.post(
                "/tts/generate", headers={"X-TTS-Session-ID": "test"}, json={**body, **options},
            )
            assert reply.raise_for_status().content == b"test audio"
            assert state.fixed_references[LANGUAGE, speaker_id] is original
        assert path.read_text(encoding="utf-8") == content

    assert [msg.sampling_params.seed for msg in submitted] == [None, SEED, 123, SEED, SEED, SEED]
    assert all(msg.speaker_embedding is not None for msg in submitted)
    assert [msg.long_form_chunk_chars for msg in submitted] == [150] * 6
    assert [msg.long_form_window_chunks for msg in submitted] == [2, 1, 2, 2, 2, 1]
    contexts = [msg.continuation_context for msg in submitted]
    assert [context is not None for context in contexts] == [False, True, False, False, False, True]
    first, last = contexts[1], contexts[-1]
    assert first is not None and last is not None
    assert first.anchor_text == last.anchor_text == "Une voix."
    assert first.anchor_codes == last.anchor_codes == [[42] * 9]
    assert first.anchor_codes is not last.anchor_codes


@pytest.mark.parametrize(
    "invalid,error",
    [
        ("num_tokenizer", "require --num-tokenizer 0"),
        ("revision", "Set ZONOS2_MODEL_REVISION"),
        ("file", "No such file"),
    ],
)
def test_invalid_fixed_reference_config_fails_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invalid: str, error: str,
) -> None:
    path, _, data = _asset(tmp_path)
    if invalid != "file":
        path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setenv("ZONOS2_FIXED_REFERENCE_FILE", str(path))
    if invalid == "revision":
        monkeypatch.delenv("ZONOS2_MODEL_REVISION", raising=False)
    else:
        monkeypatch.setenv("ZONOS2_MODEL_REVISION", REVISION)
    state = _frontend(tmp_path, monkeypatch, num_tokenizer=int(invalid == "num_tokenizer"))
    with pytest.raises((ValueError, FileNotFoundError), match=error), TestClient(api_server.app):
        pytest.fail("The server accepted startup with an invalid reference setup.")
    assert not state.fixed_references
