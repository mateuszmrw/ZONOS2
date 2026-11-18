from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from zonos2.core import TTSReq
from zonos2.core import TTSSamplingParams as RuntimeSamplingParams
from zonos2.message import BaseBackendMsg, BatchTTSTokenizerMsg, TTSSamplingParams, TTSUserMsg
from zonos2.message.tts import TTSCancelMsg, TTSDetokenizeMsg
from zonos2.scheduler import scheduler as scheduler_module
from zonos2.scheduler.decode import TTSDecodeManager
from zonos2.scheduler.scheduler import TTSForwardData, TTSScheduler


@pytest.fixture(autouse=True)
def disable_rank_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("debug_rank0", "warning_rank0", "info_rank0"):
        monkeypatch.setattr(scheduler_module.logger, name, lambda *args: None)


def _req(slot: int, uid: int) -> TTSReq:
    return TTSReq(torch.zeros((2, 10)), slot, 1, 100, uid, RuntimeSamplingParams(), cast(Any, None))


def _scheduler() -> Any:
    return SimpleNamespace(
        engine=SimpleNamespace(max_seq_len=16),
        waiting_reqs=[],
        decode_manager=TTSDecodeManager(),
        finished_reqs=set(),
        sent=[],
    )


def test_rejects_oversized_prompt_with_a_terminal_reply() -> None:
    scheduler = _scheduler()
    scheduler.send_result = scheduler.sent.append
    scheduler._reject = lambda *args: TTSScheduler._reject(scheduler, *args)
    req = TTSUserMsg(8, torch.zeros((16, 10)), TTSSamplingParams())
    TTSScheduler._process_one_msg(scheduler, req)
    assert not scheduler.waiting_reqs
    (batch,) = scheduler.sent
    assert isinstance(batch, BatchTTSTokenizerMsg)
    (reply,) = batch.data
    assert isinstance(reply, TTSDetokenizeMsg)
    assert reply.uid == 8 and reply.finished and reply.finish_reason == "failed"
    assert reply.error and "context" in reply.error


@pytest.mark.parametrize("waiting_only", [False, True])
def test_cancel_clears_waiting_or_running_cfg_pair(waiting_only: bool) -> None:
    scheduler = _scheduler()
    scheduler.waiting_reqs = [
        TTSUserMsg(uid, torch.zeros((2, 10)), TTSSamplingParams()) for uid in (1, 2)
    ]
    pair = [_req(0, 1), _req(1, 1)]
    sibling = _req(2, 2)
    if not waiting_only:
        scheduler.decode_manager.add_reqs([*pair, sibling])
    scheduler._cancel = lambda uid, last_data: TTSScheduler._cancel(scheduler, uid, last_data)
    cancel = TTSCancelMsg(1)
    # The same decoder is used on the primary scheduler and on its TP ranks.
    decoded: object = BaseBackendMsg.decoder(cancel.encoder())
    assert isinstance(decoded, TTSCancelMsg)
    TTSScheduler._process_one_msg(scheduler, decoded)
    assert [req.uid for req in scheduler.waiting_reqs] == [2]
    assert scheduler.finished_reqs == (set() if waiting_only else set(pair))
    assert scheduler.decode_manager.running_reqs == (set() if waiting_only else {sibling})


def test_decode_order_uses_table_slots_not_object_identity() -> None:
    decode = TTSDecodeManager()
    decode.add_reqs(_req(slot, slot) for slot in reversed(range(8)))
    batch = decode.schedule_next_batch()
    assert batch is not None
    assert [req.table_idx for req in batch.reqs] == list(range(8))


def test_cancel_waits_for_inflight_cuda_result_before_freeing_slots() -> None:
    scheduler = _scheduler()
    scheduler.n_codebooks = 9
    in_flight = _req(1, 1)
    scheduler.send_result = scheduler.sent.append
    released = []

    class Event:
        def synchronize(self) -> None:
            released.append("synced")

    batch = SimpleNamespace(reqs=[in_flight], size=1, padded_size=1)
    pending = cast(
        TTSForwardData, (SimpleNamespace(batch=batch), (None, torch.zeros((1, 10)), Event())),
    )
    TTSScheduler._cancel(scheduler, 1, pending)
    assert scheduler.finished_reqs == {in_flight}
    assert not released
    scheduler.page_table = torch.arange(12).view(3, 4)
    scheduler.table_manager = SimpleNamespace(free=lambda index: released.append(index))
    scheduler.cache_manager = SimpleNamespace(
        free_slots=lambda slots: released.extend(slots.tolist()),
        free_handle=lambda _: released.append("handle"),
    )
    scheduler._release_finished = lambda reqs: TTSScheduler._release_finished(scheduler, reqs)
    TTSScheduler._process_last_data(scheduler, pending, None)
    assert released == ["synced", 4, "handle", 1]
    assert not scheduler.finished_reqs
    assert not scheduler.sent[0].data  # No audio from the abandoned forward pass.


@pytest.mark.parametrize("eos", [False, True])
def test_overlapping_forward_accounts_for_the_frame_being_returned(eos: bool) -> None:
    scheduler = _scheduler()
    scheduler.engine.max_seq_len = 64
    scheduler.n_codebooks = 9
    scheduler.send_result = scheduler.sent.append
    scheduler._release_finished = lambda _: None
    limit = 16
    req = TTSReq(
        torch.zeros((2, 10)), 0, 0, limit, 1, RuntimeSamplingParams(), cast(Any, None),
    )
    for _ in range(limit):
        req.complete_one()
    frame = torch.zeros((1, 10), dtype=torch.int32)
    if eos:
        frame[0, 1] = req.eoa_id
    batch = SimpleNamespace(reqs=[req], size=1, padded_size=1)
    ack = cast(TTSForwardData, (
        SimpleNamespace(batch=batch), (None, frame, SimpleNamespace(synchronize=lambda: None)),
    ))
    # The device already computed frame 16, but only frame 15 is being returned.
    TTSScheduler._process_last_data(scheduler, ack, ack)
    result = scheduler.sent[-1].data[0]
    assert not result.finished
    assert result.eos_frame == (13 if eos else None)
    TTSScheduler._process_last_data(scheduler, ack, None)
    assert scheduler.sent[-1].data[0].finish_reason == "frame_limit"
