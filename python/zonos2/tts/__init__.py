from __future__ import annotations

from typing import TYPE_CHECKING

from .prompt import (
    BYTE_TEXT_VOCAB_SIZE,
    TTSPromptBuilder,
    TTSPromptConfig,
    accurate_mode_token_id,
    byte_text_vocab_size,
    conditioned_text_vocab_size,
    quality_token_id,
    speaker_background_token_id,
    speaking_rate_token_id,
    text_to_byte_ids,
    text_to_prompt_tokens,
    tokens_to_prompt_tokens,
)
from .sequence import TTSSequence

if TYPE_CHECKING:
    from .llm import TTSLLM


def __getattr__(name: str) -> type[TTSLLM]:
    # Prompt-only and server-side CPU utilities do not need the CUDA inference stack.
    if name == "TTSLLM":
        from .llm import TTSLLM

        return TTSLLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BYTE_TEXT_VOCAB_SIZE",
    "TTSPromptBuilder",
    "TTSPromptConfig",
    "TTSSequence",
    "accurate_mode_token_id",
    "byte_text_vocab_size",
    "conditioned_text_vocab_size",
    "quality_token_id",
    "speaker_background_token_id",
    "speaking_rate_token_id",
    "text_to_byte_ids",
    "text_to_prompt_tokens",
    "tokens_to_prompt_tokens",
    "TTSLLM",
]
