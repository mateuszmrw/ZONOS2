"""Load read-only, pre-generated speech prompts for registered voices."""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
AudioCode = Annotated[int, Field(ge=0, le=1025)]


class FixedReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    speaker_file: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]+\.npy$")]
    speaker_sha256: Digest
    language: str = Field(min_length=1, max_length=16)
    seed: int = Field(ge=0, le=2**32 - 1)
    text: str = Field(min_length=1, max_length=512)
    speech_frames: int = Field(gt=0)
    prompt_codes: tuple[tuple[AudioCode, ...], ...] = Field(min_length=1, max_length=1536)


class FixedReferenceAsset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    version: Literal[1]
    model_revision: Revision
    params_sha256: Digest
    references: tuple[FixedReference, ...] = Field(min_length=1, max_length=32)


def select_fixed_reference(
    references: Mapping[tuple[str, str], FixedReference],
    *,
    language: str,
    speaker_id: str | None,
    seed: int | None,
) -> FixedReference | None:
    if speaker_id is None or seed is None:
        return None
    reference = references.get((language, speaker_id))
    return reference if reference is not None and reference.seed == seed else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(64 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_fixed_references(
    path: Path,
    *,
    model_dir: Path,
    model_revision: str,
    n_codebooks: int,
    max_seq_len: int,
    languages: Collection[str],
    voices: Collection[tuple[str, Path]],
) -> Mapping[tuple[str, str], FixedReference]:
    """Validate a trusted local asset against the model config and registered `.npy` files."""
    with path.open("rb") as file:
        payload = file.read(512 * 1024 + 1)
    if len(payload) > 512 * 1024:
        raise ValueError("Fixed reference asset exceeds 512 KiB.")
    asset = FixedReferenceAsset.model_validate_json(payload)
    if (
        asset.model_revision != model_revision
        or asset.params_sha256 != _sha256(model_dir / "params.json")
    ):
        raise ValueError("Fixed references do not match the configured ZONOS2 model.")

    result: dict[tuple[str, str], FixedReference] = {}
    for ref in asset.references:
        if ref.language not in languages:
            raise ValueError(f"Unsupported fixed reference language: {ref.language!r}.")
        matches = [(speaker_id, file) for speaker_id, file in voices if file.name == ref.speaker_file]
        if len(matches) != 1 or _sha256(matches[0][1]) != ref.speaker_sha256:
            raise ValueError(f"Fixed reference does not match one speaker file {ref.speaker_file!r}.")
        key = (ref.language, matches[0][0])
        if key in result:
            raise ValueError(f"Duplicate fixed reference for {ref.language} and {ref.speaker_file!r}.")
        if (
            not ref.text.strip()
            or ref.speech_frames > len(ref.prompt_codes)
            or any(len(row) != n_codebooks for row in ref.prompt_codes)
            or len(ref.prompt_codes) + len(ref.text.encode("utf-8")) + 8 >= max_seq_len
        ):
            raise ValueError(f"Invalid fixed reference prompt for {ref.language}.")
        result[key] = ref
    return MappingProxyType(result)
