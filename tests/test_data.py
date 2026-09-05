import torch

from qera_exp.data import _windowize


def test_random_blocks_are_non_overlapping_and_deterministic() -> None:
    stream = torch.arange(80)
    spec = {"sequence_length": 8, "windows": 4, "sampling": "random_blocks", "seed": 17}
    first, first_rows = _windowize(stream, spec)
    second, second_rows = _windowize(stream, spec)
    assert torch.equal(first, second)
    assert first_rows == second_rows
    assert len({row["source_block_index"] for row in first_rows}) == 4
    assert [row["source_block_index"] for row in first_rows] != [0, 1, 2, 3]


def test_all_uses_every_complete_window() -> None:
    ids, rows = _windowize(torch.arange(19), {"sequence_length": 4, "windows": "all", "sampling": "prefix"})
    assert ids.shape == (4, 4)
    assert len(rows) == 4
