import torch

from qera_exp.statistics import exact_ce_hidden_gradient


def test_exact_ce_gradient_respects_attention_mask() -> None:
    generator = torch.Generator().manual_seed(31)
    hidden = torch.randn(1, 6, 5, generator=generator, dtype=torch.float32)
    output_weight = torch.randn(17, 5, generator=generator, dtype=torch.float32)
    ids = torch.randint(0, 17, (1, 6), generator=generator)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]])
    gradient, nll, count = exact_ce_hidden_gradient(hidden, output_weight, ids, mask, chunk_size=2)
    assert count == 3
    assert nll > 0
    assert torch.count_nonzero(gradient[:, 3:]) == 0
