import torch
from torch import nn

from qera_exp.modeling import discover_target_modules


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        self.v_proj = nn.Linear(4, 2, bias=False)
        self.o_proj = nn.Linear(4, 4, bias=False)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 8, bias=False)
        self.up_proj = nn.Linear(4, 8, bias=False)
        self.down_proj = nn.Linear(8, 4, bias=False)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = Attention()
        self.mlp = MLP()


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Decoder()


def test_discovers_qwen_llama_projection_names() -> None:
    suffixes = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    config = {"model": {"target_suffixes": suffixes, "layers": [1]}}
    names = discover_target_modules(FakeModel(), config)
    assert len(names) == 7
    assert all(name.startswith("model.layers.1.") for name in names)
    assert names[0].endswith("self_attn.q_proj")
    assert names[-1].endswith("mlp.down_proj")
