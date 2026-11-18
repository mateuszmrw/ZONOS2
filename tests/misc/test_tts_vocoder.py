from __future__ import annotations

import numpy as np
import pytest
import torch
from zonos2.message import TTSDetokenizeMsg
from zonos2.tokenizer import vocoder


def test_vocoder_does_not_send_eos_frame_or_after_to_dac(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_codes: list[torch.Tensor] = []

    def fake_decode_dac(codes: torch.Tensor) -> torch.Tensor:
        captured_codes.append(codes.cpu().clone())
        return torch.zeros((codes.shape[0], codes.shape[1]), dtype=torch.float32)

    monkeypatch.setattr(vocoder, "decode_dac", fake_decode_dac)

    manager = vocoder.TTSVocoderManager(
        n_codebooks=3,
        audio_pad_id=1025,
        min_decode_chunk=99,
        overlap_frames=0,
        hop_length=1,
    )
    chunks = manager.decode_frames(
        [
            TTSDetokenizeMsg(uid=1, audio_codes=[10, 20, 30], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[11, 21, 31], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[999, 22, 32], finished=False),
            TTSDetokenizeMsg(uid=1, audio_codes=[13, 999, 33], finished=False),
            TTSDetokenizeMsg(
                uid=1,
                audio_codes=[14, 24, 999],
                finished=True,
                eos_frame=2,
            ),
        ]
    )

    assert chunks[:-1] == [b"", b"", b"", b""]
    assert chunks[-1]
    assert len(captured_codes) == 1
    assert captured_codes[0].tolist() == [
        [
            [10, 21, 32],
            [11, 22, 33],
        ]
    ]


@pytest.mark.parametrize(
    "rows, eos_frame, expected_frames",
    [(1, None, 0), (12, None, 4), (12, 6, 4), (12, 2, 2), (25, None, 17)],
)
def test_frame_limit_decodes_only_aligned_frames(
    monkeypatch: pytest.MonkeyPatch, rows: int, eos_frame: int | None, expected_frames: int
) -> None:
    def fake_decode_dac(codes: torch.Tensor) -> torch.Tensor:
        assert not (codes == 1025).any()
        return codes.sum(dim=-1, dtype=torch.float32).cpu()

    monkeypatch.setattr(vocoder, "decode_dac", fake_decode_dac)
    manager = vocoder.TTSVocoderManager(hop_length=1)
    generated = [[row * 9 + cb for cb in range(9)] for row in range(rows)]
    chunks = manager.decode_frames([
        TTSDetokenizeMsg(
            uid=1, audio_codes=row, finished=i == rows - 1,
            eos_frame=eos_frame if i == rows - 1 else None,
            finish_reason="frame_limit" if i == rows - 1 else None,
        )
        for i, row in enumerate(generated)
    ])
    pcm = np.frombuffer(b"".join(chunks), dtype=np.float32)
    expected = [
        sum(generated[row + cb][cb] for cb in range(9))
        for row in range(expected_frames)
    ]
    np.testing.assert_allclose(pcm, expected, rtol=0, atol=0.001)


def test_primed_vocoder_flushes_only_new_samples_across_all_codebooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Give each raw row and codebook a distinct value; reducing CB0 alone
    # would miss incorrect delayed-codebook alignment at the end of a response.
    def fake_decode_dac(codes: torch.Tensor) -> torch.Tensor:
        return codes.sum(dim=-1, dtype=torch.float32).cpu()

    monkeypatch.setattr(vocoder, "decode_dac", fake_decode_dac)
    manager = vocoder.TTSVocoderManager(
        n_codebooks=9, min_decode_chunk=16, overlap_frames=4, hop_length=1
    )
    uid = 1
    prefix = [[row * 9 + cb for cb in range(9)] for row in range(20)]
    generated = [[200 + row * 9 + cb for cb in range(9)] for row in range(50)]
    manager.prime(uid, prefix)
    chunks = manager.decode_frames(
        [TTSDetokenizeMsg(uid=uid, audio_codes=row, finished=False) for row in generated[:42]]
    )
    pcm = np.frombuffer(
        b"".join(chunks) + manager.finish(uid, lookahead=generated[42:]), dtype=np.float32
    )
    expected = [
        sum(generated[row + cb][cb] for cb in range(9))
        for row in range(42)
    ]
    np.testing.assert_allclose(pcm, expected, rtol=0, atol=0.001)
    assert manager.finish(uid) == b""


@pytest.mark.parametrize("lengths", [(1, 2, 71), (27, 48, 19), (109, 7, 223)])
def test_deferred_decode_is_identical_to_one_continuous_decode(
    monkeypatch: pytest.MonkeyPatch, lengths: tuple[int, ...]
) -> None:
    def fake_decode(codes: torch.Tensor) -> torch.Tensor:
        # Use all codebooks and actual neighboring audio frames, like a noncausal decoder.
        signal = codes.sum(dim=-1).float()
        padded = torch.nn.functional.pad(signal, (1, 1))
        return (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]).repeat_interleave(
            2, dim=1
        ).cpu()

    monkeypatch.setattr(vocoder, "decode_dac", fake_decode)
    manager = vocoder.TTSVocoderManager(hop_length=2)
    rows = [[(17 * i + 19 * j) % 1024 for j in range(9)] for i in range(sum(lengths) + 8)]

    def feed(uid: int, frames: list[list[int]]) -> bytes:
        return b"".join(manager.decode_frames([TTSDetokenizeMsg(uid, row, False) for row in frames]))

    continuous = feed(0, rows[:-8]) + manager.finish(0, lookahead=rows[-8:])
    offset = 0
    parts = []
    tail = None
    for uid, length in enumerate(lengths, 1):
        if tail is not None:
            manager.resume(uid, tail)
        parts.append(feed(uid, rows[offset:offset + length]))
        offset += length
        tail = manager.suspend(uid, lookahead=rows[offset:offset + 8])
        assert len(tail.frames) <= manager.overlap_frames + manager.min_decode_chunk + 7
        assert manager.finish(uid) == b""
    assert tail is not None
    # The same tail can be retried on a new UID without advancing the snapshot.
    manager.resume(10, tail)
    final = manager.finish(10, lookahead=tail.lookahead)
    manager.resume(11, tail)
    assert manager.finish(11, lookahead=tail.lookahead) == final
    assert b"".join(parts) + final == continuous
    assert len(continuous) == 4 * 2 * sum(lengths)
