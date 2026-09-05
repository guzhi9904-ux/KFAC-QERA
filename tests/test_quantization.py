import torch

from qera_exp.quantization import quantize_mxint4, reconstruct


def test_mxint4_reconstructs_codes_bit_identically() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(13, 97, generator=generator, dtype=torch.float32)
    result = quantize_mxint4(weight, block_size=32)
    rebuilt = reconstruct(result.codes, result.exponents, weight.shape[1], block_size=32)
    assert torch.equal(result.reconstructed, rebuilt)
    assert result.codes.dtype == torch.int8
    assert result.exponents.dtype == torch.uint8
    assert int(result.codes.abs().max()) <= 7
    assert result.padding == 31


def test_zero_blocks_are_finite() -> None:
    result = quantize_mxint4(torch.zeros(3, 64), block_size=32)
    assert torch.equal(result.reconstructed, torch.zeros(3, 64))
    assert torch.isfinite(result.reconstructed).all()
