from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn.functional as F
from zonos2.utils import init_logger

logger = init_logger(__name__)

EOA_TOKEN = 1024  # End of audio token


def _apply_top_p(probs: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Apply nucleus (top-p) sampling to probabilities."""
    enabled = (thresholds > 0) & (thresholds < 1)
    if not bool(enabled.any()):
        return probs
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = enabled & (probs_sum - probs_sort > thresholds)
    probs_sort = probs_sort.masked_fill(mask, 0.0)
    filtered = probs.scatter(-1, probs_idx, probs_sort)
    return torch.where(enabled, filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8), probs)


def _apply_min_p(probs: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    """Apply min-p filtering to probabilities."""
    enabled = thresholds > 0
    if not bool(enabled.any()):
        return probs
    top_probs, _ = probs.max(dim=-1, keepdim=True)
    filtered = probs.masked_fill(probs < (thresholds * top_probs), 0.0)
    return torch.where(enabled, filtered / filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8), probs)


def apply_repetition_penalty(
    logits: torch.Tensor,
    repetition_token_ids: torch.Tensor | None,
    repetition_penalties: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a per-sequence repetition penalty to per-codebook logits.

    repetition_token_ids is shaped (B, n_codebooks, window). A token id in one
    codebook only penalizes the matching codebook's logits.
    """
    if repetition_token_ids is None or repetition_penalties is None:
        return logits
    if repetition_token_ids.numel() == 0:
        return logits

    B, C, V = logits.shape
    safe_token_ids = repetition_token_ids.clamp(min=0, max=V - 1).long()
    valid = (repetition_token_ids >= 0) & (repetition_token_ids < V)

    counts = torch.zeros((B, C, V), dtype=torch.int32, device=logits.device)
    counts.scatter_add_(-1, safe_token_ids, valid.to(torch.int32))
    repeated = counts > 0

    penalties = repetition_penalties.view(B, 1, 1).clamp(min=1.0)
    adjusted_logits = torch.where(logits > 0, logits / penalties, logits * penalties)
    return torch.where(repeated, adjusted_logits, logits)


_debug_sample_count = 0


def _compute_logit_stats_debug(logits: torch.Tensor) -> dict:
    """Compute per-codebook stats matching reference format.

    Args:
        logits: (B, n_codebooks, vocab) - uses first batch element

    Returns:
        Dict with per-codebook stats: entropy, top1_prob, margin, top5_ids
    """
    lg = logits[0].float()  # (n_codebooks, vocab)

    probs = F.softmax(lg, dim=-1)
    log_probs = F.log_softmax(lg, dim=-1)

    # Entropy: -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)

    # Top-k probs and ids
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    top1_prob = sorted_probs[:, 0]
    top5_ids = sorted_indices[:, :5]

    # Logit margin (top1 - top2)
    sorted_logits, _ = lg.sort(dim=-1, descending=True)
    margin = sorted_logits[:, 0] - sorted_logits[:, 1]

    return {
        "entropy": entropy.tolist(),
        "top1_prob": top1_prob.tolist(),
        "margin": margin.tolist(),
        "top5_ids": top5_ids.tolist(),
    }


def sample_tts(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    repetition_token_ids: torch.Tensor | None = None,
    repetition_penalties: torch.Tensor | None = None,
    text_vocab: int = 0,
    generators: list[torch.Generator | None] | None = None,
) -> List[List[int]]:
    """Functional interface for TTS sampling.

    Args:
        logits: Shape (B, n_codebooks, vocab_size)
        temperatures: Per-sequence temperatures (B,)
        top_ks: Per-sequence top-k values (B,)
        top_ps: Per-sequence top-p values (B,)
        min_ps: Per-sequence min-p values (B,)
        repetition_token_ids: Recent token ids per sequence/codebook (B, C, W)
        repetition_penalties: Per-sequence repetition penalties (B,), 1.0 disables
        text_vocab: Text vocabulary size (appended as placeholder)

    Returns:
        List of unpacked tokens, each is [cb0, cb1, ..., cb8, text_placeholder]
    """
    global _debug_sample_count
    B, C, V = logits.shape

    # Debug: print detailed stats matching reference format for first 10 samples
    # Guard with is_current_stream_capturing to avoid CUDA graph capture errors
    if (logger.isEnabledFor(logging.DEBUG)
            and _debug_sample_count < 10
            and EOA_TOKEN < V
            and not (logits.is_cuda and torch.cuda.is_current_stream_capturing())):
        _debug_sample_count += 1
        step = _debug_sample_count

        stats = _compute_logit_stats_debug(logits)

        logger.debug("Step %d per-codebook stats:", step)
        for cb in range(C):
            top5 = stats["top5_ids"][cb]
            logger.debug(
                "  CB%d: entropy=%.4f, top1_prob=%.4f, top1_id=%d, margin=%.2f, top5=%s",
                cb, stats["entropy"][cb], stats["top1_prob"][cb],
                top5[0], stats["margin"][cb], top5,
            )

        probs_raw = F.softmax(logits[0], dim=-1)
        eoa_probs = probs_raw[:, EOA_TOKEN].tolist()
        eoa_logits = logits[0, :, EOA_TOKEN].tolist()
        logger.debug(
            "Step %d EOA(1024) logits: [%s]",
            step, ", ".join(f"{x:.2f}" for x in eoa_logits),
        )
        logger.debug(
            "Step %d EOA(1024) probs:  [%s]",
            step, ", ".join(f"{x:.4f}" for x in eoa_probs),
        )

    logits = apply_repetition_penalty(
        logits,
        repetition_token_ids=repetition_token_ids,
        repetition_penalties=repetition_penalties,
    )

    greedy = (temperatures <= 0).tolist()
    if all(greedy):
        next_ids = torch.argmax(logits, dim=-1)
    else:
        temp_broadcast = temperatures.view(B, 1, 1).clamp(min=1e-8)
        scaled = logits / temp_broadcast
        # One topk operation, with a separate threshold for each request.
        ks = torch.where((top_ks > 0) & (top_ks < V), top_ks, 0)
        largest_k = int(ks.max().item())
        if largest_k:
            values = torch.topk(scaled, largest_k, dim=-1).values
            indices = (ks.long().view(B, 1, 1) - 1).clamp_min(0).expand(B, C, 1)
            kth = values.gather(-1, indices)
            scaled = scaled.masked_fill((ks.view(B, 1, 1) > 0) & (scaled < kth), -torch.inf)

        probs = F.softmax(scaled, dim=-1)
        probs = _apply_top_p(probs, top_ps.view(B, 1, 1))
        probs = _apply_min_p(probs, min_ps.view(B, 1, 1))

        # An aggressive min_p (>1) or top_p can remove every token in a row,
        # leaving an all-zero distribution. torch.multinomial would then raise a
        # CUDA device-side assert that aborts the scheduler, so fall back to
        # greedy (argmax) for any fully-masked row.
        invalid_rows = ~(probs.sum(dim=-1) > 0)
        if bool(invalid_rows.any()):
            fallback = torch.zeros_like(probs).scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
            probs = torch.where(invalid_rows.unsqueeze(-1), fallback, probs)

        # Use per-request generators for deterministic sampling when seeds are set
        if any(greedy) or (generators and any(g is not None for g in generators)):
            rows = []
            for i in range(B):
                gen = generators[i] if generators and i < len(generators) else None
                rows.append(logits[i].argmax(dim=-1) if greedy[i] else
                            torch.multinomial(probs[i], num_samples=1, generator=gen).squeeze(-1))
            next_ids = torch.stack(rows)
        else:
            next_ids = torch.multinomial(probs.view(B * C, V), num_samples=1).view(B, C)

    # Return unpacked format: list of (audio codes + text placeholder)
    next_ids_list = next_ids.tolist()

    # Debug: print sampled tokens for first 10 samples
    if (logger.isEnabledFor(logging.DEBUG)
            and _debug_sample_count <= 10
            and not (logits.is_cuda and torch.cuda.is_current_stream_capturing())):
        step = _debug_sample_count
        sampled = next_ids_list[0] if next_ids_list else []
        logger.debug("Step %d sampled: %s", step, sampled)
        for i, row in enumerate(next_ids_list):
            eoa_positions = [j for j, t in enumerate(row) if t == EOA_TOKEN]
            if eoa_positions:
                logger.debug(
                    "Step %d *** EOA SAMPLED at CB %s for batch %d ***",
                    step, eoa_positions, i,
                )

    return [row + [text_vocab] for row in next_ids_list]
