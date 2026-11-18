import pytest
import torch
from zonos2.tts.sampler import sample_tts


@pytest.mark.parametrize(
    "top_ks, top_ps, min_ps",
    [
        ([1, 0, 3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([0, 0, 0], [0.08, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ([0, 0, 0], [0.0, 0.0, 0.0], [0.99, 0.0, 1.1]),
    ],
)
def test_mixed_sampling_settings_match_each_request_alone(
    top_ks: list[int], top_ps: list[float], min_ps: list[float]
) -> None:
    # Enough draws to distinguish the almost-flat distribution from greedy.
    logits = torch.tensor([1.0, 0.95, 0.9, 0.85]).expand(3, 32, 4)
    temperatures = torch.tensor([0.9, 1.5, 1.0])
    ks, ps, mins = torch.tensor(top_ks), torch.tensor(top_ps), torch.tensor(min_ps)
    batch = sample_tts(
        logits, temperatures, ks, ps, mins,
        generators=[torch.Generator().manual_seed(s) for s in (50, 51, 52)],
    )
    assert set(batch[1][:-1]) == {0, 1, 2, 3}
    for i, seed in enumerate((50, 51, 52)):
        single = sample_tts(
            logits[i:i + 1],
            temperatures[i:i + 1], ks[i:i + 1], ps[i:i + 1], mins[i:i + 1],
            generators=[torch.Generator().manual_seed(seed)],
        )
        assert batch[i] == single[0]


def test_mixed_greedy_request_does_not_draw_from_its_generator() -> None:
    greedy = torch.Generator().manual_seed(50)
    state = greedy.get_state().clone()
    logits = torch.zeros((2, 16, 4))
    rows = sample_tts(
        logits, torch.tensor([0.0, 1.0]), torch.tensor([0, 0]),
        torch.zeros(2), torch.zeros(2), generators=[greedy, torch.Generator().manual_seed(51)],
    )
    assert rows[0][:-1] == [0] * 16
    assert torch.equal(state, greedy.get_state())
