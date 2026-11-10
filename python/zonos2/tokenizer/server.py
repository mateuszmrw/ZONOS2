from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass, field, replace
from typing import List

import torch
from zonos2.message import (
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchTokenizerMsg,
    BatchTTSBackendMsg,
    BatchTTSFrontendMsg,
    BatchTTSTokenizerMsg,
    TTSAudioReply,
    TTSDetokenizeMsg,
    TTSTokenizeMsg,
    TTSUserMsg,
)
from zonos2.message.tts import TTSContinuationContext, TTSSamplingParams
from zonos2.tts.longform import (
    build_continuation_prompt,
    context_chunks,
    split_text,
    trim_chunk_codes,
)
from zonos2.tts.prompt import TTSPromptBuilder, TTSPromptConfig
from zonos2.utils import ZmqPullQueue, ZmqPushQueue, init_logger


@dataclass
class _LongFormState:
    """Per-request state driving windowed teacher-forced continuation.

    Chunks are generated sequentially; each chunk after the first continues a
    window of WHOLE prior chunks fed back as an acoustic prefix, with their text
    in the conditioning so the prefix stays aligned. The window is the previous
    ``context_chunks`` chunks (``history_codes``/``history_text``); when
    ``pin_anchor`` is set, the first chunk (``anchor_codes``/``anchor_text``) is
    also pinned in and the middle is evicted, re-grounding every chunk on the same
    reference to prevent timbre drift.

    Generated frames stream to the client through one continuous vocoder buffer
    (keyed by the request uid), with ``finished`` withheld until the final chunk
    so there are no boundary artifacts. ``committed`` tracks how many of the
    current chunk's frames have already been fed to the vocoder (we hold back a
    tail margin while streaming so EOA/post-content frames are never decoded).
    """

    chunks: List[str]
    sampling_params: TTSSamplingParams
    speaking_rate_bucket: int | None
    final_quality: List[int | None] | None
    nonfinal_quality: List[int | None] | None
    speaker_embedding: torch.Tensor | None
    speaker_emotion_delta: torch.Tensor | None
    speaker_slot: torch.Tensor | None
    clean_speaker_background: bool
    accurate_mode: bool
    context_chunks: int = 1
    pin_anchor: bool = True
    anchor_text: str = ""
    anchor_codes: List[List[int]] = field(default_factory=list)
    index: int = 0
    history_text: List[str] = field(default_factory=list)
    history_codes: List[List[List[int]]] = field(default_factory=list)
    raw: List[List[int]] = field(default_factory=list)
    committed: int = 0
    prior_chunks: int = 0
    retain_continuation: bool = False
    valid_context: bool = True
    max_prompt_frames: int | None = None

    @property
    def is_last(self) -> bool:
        return self.index == len(self.chunks) - 1

    def accept_chunk(self, kept: List[List[int]], eos_frame: int | None) -> _LongFormState:
        first = self.prior_chunks + self.index == 0
        return replace(
            self,
            anchor_text=self.chunks[self.index] if first and self.pin_anchor else self.anchor_text,
            anchor_codes=kept if first and self.pin_anchor else self.anchor_codes,
            history_text=(
                [*self.history_text, self.chunks[self.index]][-self.context_chunks:]
                if self.context_chunks else []
            ),
            history_codes=(
                [*self.history_codes, kept][-self.context_chunks:]
                if self.context_chunks else []
            ),
            valid_context=self.valid_context and eos_frame is not None and bool(kept),
        )

    def snapshot(self) -> TTSContinuationContext | None:
        if not self.valid_context:
            return None
        return TTSContinuationContext(
            completed_chunks=self.prior_chunks + self.index + 1,
            anchor_text=self.anchor_text,
            anchor_codes=self.anchor_codes,
            history_text=self.history_text,
            history_codes=self.history_codes,
        )


# Extra frames trimmed off a non-final chunk to remove residual trailing silence.
_LONGFORM_BOUNDARY_TRIM = 3


def _unwrap_msg(
    msg: BaseTokenizerMsg | BatchTokenizerMsg | BatchTTSTokenizerMsg,
) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        return msg.data
    if isinstance(msg, BatchTTSTokenizerMsg):
        return msg.data
    return [msg]


@torch.inference_mode()
def tokenize_worker(
    *,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    ack_queue: mp.Queue[str] | None = None,
    n_codebooks: int = 9,
    audio_pad_id: int = 1025,
    text_vocab: int | None = None,
    speaking_rate_num_buckets: int = 0,
    quality_bucket_counts: List[int] | None = None,
    quality_features: List[str] | None = None,
    speaker_background_num_buckets: int = 0,
    accurate_mode_num_buckets: int = 0,
) -> None:
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    assert local_bs > 0
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    from .textnorm import TTSTextNormalizer, normalization_enabled
    from .vocoder import TTSVocoderManager

    if text_vocab is None:
        raise ValueError("TTS mode requires text_vocab from the model config.")
    textnorm_enabled = normalization_enabled()
    text_normalizer = TTSTextNormalizer() if textnorm_enabled else None
    if text_normalizer is not None:
        # Compile/load the English grammars up front so the first request does
        # not pay the one-time WFST construction cost.
        threading.Thread(
            target=text_normalizer.warmup, args=(["en"],), daemon=True
        ).start()
    tts_prompt_builder = TTSPromptBuilder(
        TTSPromptConfig(
            n_codebooks=n_codebooks,
            audio_pad_id=audio_pad_id,
            text_vocab=text_vocab,
            speaking_rate_num_buckets=speaking_rate_num_buckets,
            quality_bucket_counts=tuple(quality_bucket_counts or ()),
            speaker_background_num_buckets=speaker_background_num_buckets,
            accurate_mode_num_buckets=accurate_mode_num_buckets,
            prepend_silence=True,
        )
    )
    tts_vocoder_manager = TTSVocoderManager(
        n_codebooks=n_codebooks,
        audio_pad_id=audio_pad_id,
    )

    # ----- Long-form (windowed teacher-forced continuation) support -----
    longform_states: dict[int, _LongFormState] = {}
    # Frames held back from the vocoder while streaming a chunk so the EOA
    # countdown / end-of-content frames are never decoded before we know the trim
    # point. Must be >= (total - eos_frame) + boundary trim so streamed frames
    # never exceed the eventual trimmed length (else the vocoder buffer would
    # diverge from the next chunk's prefix at the join). total - eos_frame is at
    # most (eos alignment shift < n_codebooks) + (EOA countdown = n_codebooks + 1).
    stream_margin = 2 * n_codebooks + _LONGFORM_BOUNDARY_TRIM + 1
    trailing_silence_idx = None
    if quality_features is not None:
        for idx, feature in enumerate(quality_features):
            if feature == "trailing_silence_s":
                trailing_silence_idx = idx
                break

    def _nonfinal_quality(resolved):
        """Resolved quality list with trailing-silence conditioning removed.

        Non-final chunks must not end utterance-finally, else the joins are
        audible. If the feature order is unknown, drop quality conditioning.
        """
        if resolved is None:
            return None
        if trailing_silence_idx is None:
            return None
        out = list(resolved)
        if trailing_silence_idx < len(out):
            out[trailing_silence_idx] = None
        return out

    def _build_fresh_chunk_prompt(state: _LongFormState, q) -> torch.Tensor:
        """Fresh (chunk-0 style) prompt for the current chunk, no acoustic prefix."""
        t = tts_prompt_builder.build(
            state.chunks[state.index],
            speaking_rate_bucket=state.speaking_rate_bucket,
            quality_buckets=q,
        )
        if state.speaker_slot is not None:
            t = torch.cat([state.speaker_slot.to(dtype=t.dtype), t], dim=0)
        return t

    def _build_chunk_user_msg(state: _LongFormState, uid: int) -> TTSUserMsg:
        q = (
            state.final_quality
            if state.is_last and not state.retain_continuation else state.nonfinal_quality
        )
        # Whole prior chunks: the rolling recent window plus, when the anchor has
        # scrolled out of it, the pinned first chunk (middle evicted).
        prepend_anchor = (
            state.pin_anchor
            and bool(state.anchor_codes)
            and state.prior_chunks + state.index > state.context_chunks
        )
        texts = ([state.anchor_text] if prepend_anchor else []) + list(state.history_text)
        code_chunks = (
            [state.anchor_codes] if prepend_anchor else []
        ) + list(state.history_codes)
        context_text = " ".join(texts)
        context_codes = [f for chunk in code_chunks for f in chunk]
        cfg_uncond_input_ids = None
        if not context_codes:
            # First chunk, or teacher forcing disabled: generate fresh
            # (chunk-0 style, with the silence lead-in prefix).
            t = _build_fresh_chunk_prompt(state, q)
        else:
            t = build_continuation_prompt(
                tts_prompt_builder,
                context_text,
                state.chunks[state.index],
                context_codes,
                text_vocab=text_vocab,
                speaking_rate_bucket=state.speaking_rate_bucket,
                quality_buckets=q,
                speaker_slot=state.speaker_slot,
            )
            # Prefix CFG: the unconditional twin is the same chunk generated fresh
            # (no acoustic prefix / context text), keeping the speaker.
            if float(state.sampling_params.prefix_cfg_scale) != 1.0:
                cfg_uncond_input_ids = _build_fresh_chunk_prompt(state, q)
        marker_frames = (
            int(speaker_background_num_buckets > 0)
            + int(
                speaker_background_num_buckets > 0
                and accurate_mode_num_buckets > 0 and state.accurate_mode
            )
            if state.speaker_embedding is not None else 0
        )
        prompt_frames = max(
            len(t), len(cfg_uncond_input_ids) if cfg_uncond_input_ids is not None else 0
        )
        if (
            state.max_prompt_frames is not None
            and prompt_frames + marker_frames > state.max_prompt_frames
        ):
            raise ValueError("Continuation prompt exceeds the model context. Use smaller chunks.")
        return TTSUserMsg(
            uid=uid,
            input_ids=t,
            sampling_params=state.sampling_params,
            speaker_embedding=state.speaker_embedding,
            speaker_emotion_delta=state.speaker_emotion_delta,
            speaker_token_position=0 if state.speaker_slot is not None else -1,
            clean_speaker_background=state.clean_speaker_background,
            accurate_mode=state.accurate_mode,
            cfg_uncond_input_ids=cfg_uncond_input_ids,
        )

    def _vocode(uid: int, frames: List[List[int]], finished: bool) -> bytes:
        """Feed content frames into the continuous per-uid vocoder buffer.

        Only the final frame of the whole request carries finished=True (which
        flushes the buffer). eos_frame is never set: we only ever feed content
        frames, so no per-chunk cap is needed.
        """
        if not frames:
            if finished:
                tts_vocoder_manager.reset(uid)
            return b""
        last = len(frames) - 1
        msgs = [
            TTSDetokenizeMsg(
                uid=uid, audio_codes=f, finished=(finished and i == last)
            )
            for i, f in enumerate(frames)
        ]
        return b"".join(tts_vocoder_manager.decode_frames(msgs))

    if ack_queue is not None:
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            tts_tokenize_msg = [m for m in pending_msg if isinstance(m, TTSTokenizeMsg)]
            tts_detokenize_msg = [
                m for m in pending_msg if isinstance(m, TTSDetokenizeMsg)
            ]

            # Process TTS tokenize messages
            if len(tts_tokenize_msg) > 0:
                user_msgs = []
                for msg in tts_tokenize_msg:
                    text = msg.text
                    if text_normalizer is not None and msg.text_normalization:
                        text = text_normalizer.normalize(text, msg.language)

                    use_longform = msg.long_form
                    if use_longform is None:
                        use_longform = len(text) > msg.long_form_chunk_chars
                    chunks = (
                        split_text(
                            text, msg.long_form_chunk_chars, msg.long_form_split_mode
                        )
                        if use_longform
                        else None
                    )

                    if (
                        msg.retain_continuation or msg.continuation_context is not None
                        or (chunks is not None and len(chunks) > 1)
                    ):
                        speaker_slot = (
                            tts_prompt_builder.speaker_slot()
                            if msg.speaker_embedding is not None
                            else None
                        )
                        state = _LongFormState(
                            chunks=chunks or [text],
                            sampling_params=msg.sampling_params,
                            speaking_rate_bucket=msg.speaking_rate_bucket,
                            final_quality=msg.quality_buckets,
                            nonfinal_quality=_nonfinal_quality(msg.quality_buckets),
                            speaker_embedding=msg.speaker_embedding,
                            speaker_emotion_delta=msg.speaker_emotion_delta,
                            speaker_slot=speaker_slot,
                            clean_speaker_background=msg.clean_speaker_background,
                            accurate_mode=msg.accurate_mode,
                            context_chunks=context_chunks(
                                msg.long_form_window_chunks
                            ),
                            pin_anchor=msg.long_form_pin_anchor,
                            retain_continuation=msg.retain_continuation,
                            max_prompt_frames=msg.max_prompt_frames,
                        )
                        previous = msg.continuation_context
                        if previous is not None:
                            state = replace(
                                state,
                                prior_chunks=previous.completed_chunks,
                                anchor_text=previous.anchor_text,
                                anchor_codes=previous.anchor_codes,
                                history_text=previous.history_text,
                                history_codes=previous.history_codes,
                            )
                        try:
                            first_msg = _build_chunk_user_msg(state, msg.uid)
                        except ValueError as exc:
                            send_frontend.put(
                                TTSAudioReply(
                                    uid=msg.uid, audio_data=b"", finished=True, error=str(exc)
                                )
                            )
                            continue
                        if previous is not None and previous.history_codes:
                            tts_vocoder_manager.prime(msg.uid, previous.history_codes[-1])
                        longform_states[msg.uid] = state
                        logger.debug(
                            "Long-form uid=%d chunks=%d context=%d text='%s...'",
                            msg.uid,
                            len(state.chunks),
                            state.context_chunks,
                            msg.text[:50],
                        )
                        user_msgs.append(first_msg)
                        continue

                    t = tts_prompt_builder.build(
                        text,
                        speaking_rate_bucket=msg.speaking_rate_bucket,
                        quality_buckets=msg.quality_buckets,
                    )
                    speaker_token_position = msg.speaker_token_position
                    if msg.speaker_embedding is not None:
                        # Canonical speaker slot is token position 0, matching training.
                        speaker_slot = tts_prompt_builder.speaker_slot(
                            dtype=t.dtype,
                            device=t.device,
                        )
                        t = torch.cat([speaker_slot, t], dim=0)
                        speaker_token_position = 0
                    logger.debug(
                        "Tokenize uid=%d speaking_rate_bucket=%s quality_buckets=%s "
                        "text='%s...' frames=%d",
                        msg.uid,
                        msg.speaking_rate_bucket,
                        msg.quality_buckets,
                        msg.text[:50],
                        len(t),
                    )
                    user_msgs.append(
                        TTSUserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                            speaker_embedding=msg.speaker_embedding,
                            speaker_token_position=speaker_token_position,
                            speaker_emotion_delta=msg.speaker_emotion_delta,
                            clean_speaker_background=msg.clean_speaker_background,
                            accurate_mode=msg.accurate_mode,
                        )
                    )
                if user_msgs:
                    batch_output = BatchTTSBackendMsg(data=user_msgs)
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)

            # Process TTS detokenize (vocoder) messages
            if len(tts_detokenize_msg) > 0:
                normal_msgs = []
                replies: List[TTSAudioReply] = []
                backend_submits: List[TTSUserMsg] = []

                for msg in tts_detokenize_msg:
                    current_state = longform_states.get(msg.uid)
                    if current_state is None:
                        normal_msgs.append(msg)
                        continue
                    state = current_state

                    state.raw.append(msg.audio_codes)
                    if not msg.finished:
                        # Stream content frames, holding back a tail margin so the
                        # EOA countdown is never decoded before the trim is known.
                        feed_upto = len(state.raw) - stream_margin
                        if feed_upto > state.committed:
                            new = state.raw[state.committed:feed_upto]
                            state.committed = feed_upto
                            audio = _vocode(msg.uid, new, finished=False)
                            if audio:
                                replies.append(
                                    TTSAudioReply(
                                        uid=msg.uid, audio_data=audio, finished=False
                                    )
                                )
                        continue

                    # Chunk finished: trim to content, feed the remainder.
                    is_last = state.is_last
                    kept = trim_chunk_codes(
                        state.raw,
                        msg.eos_frame,
                        drop_trailing_frames=(
                            0 if is_last and not state.retain_continuation
                            else _LONGFORM_BOUNDARY_TRIM
                        ),
                    )
                    new = kept[state.committed:] if state.committed < len(kept) else []
                    audio = _vocode(msg.uid, new, finished=is_last)
                    state = state.accept_chunk(kept, msg.eos_frame)
                    if audio or is_last:
                        replies.append(
                            TTSAudioReply(
                                uid=msg.uid, audio_data=audio, finished=is_last,
                                continuation_context=(
                                    state.snapshot()
                                    if is_last and state.retain_continuation else None
                                ),
                                continuation_error=(
                                    "Generation ended without complete audio. Start a new chain or use smaller chunks."
                                    if is_last and state.retain_continuation and not state.valid_context
                                    else None
                                ),
                            )
                        )
                    if is_last:
                        longform_states.pop(msg.uid, None)
                    else:
                        state = replace(state, index=state.index + 1, raw=[], committed=0)
                        try:
                            next_msg = _build_chunk_user_msg(state, msg.uid)
                        except ValueError as exc:
                            longform_states.pop(msg.uid, None)
                            tts_vocoder_manager.reset(msg.uid)
                            replies.append(
                                TTSAudioReply(
                                    uid=msg.uid, audio_data=b"", finished=True, error=str(exc)
                                )
                            )
                        else:
                            longform_states[msg.uid] = state
                            backend_submits.append(next_msg)

                if normal_msgs:
                    audio_chunks = tts_vocoder_manager.decode_frames(normal_msgs)
                    for m, audio in zip(normal_msgs, audio_chunks, strict=True):
                        replies.append(
                            TTSAudioReply(
                                uid=m.uid, audio_data=audio, finished=m.finished
                            )
                        )

                if replies:
                    batch_output = BatchTTSFrontendMsg(data=replies)
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_frontend.put(batch_output)

                if backend_submits:
                    batch_output = BatchTTSBackendMsg(data=backend_submits)
                    if len(batch_output.data) == 1:
                        batch_output = batch_output.data[0]
                    send_backend.put(batch_output)

    except KeyboardInterrupt:
        pass
