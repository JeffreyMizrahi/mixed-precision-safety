from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch
from torch import nn


def fake_quantize(weight: torch.Tensor, bits: int) -> torch.Tensor:
    if bits < 2 or bits > 16:
        raise ValueError(f"bits must be in [2, 16], got {bits}")

    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))

    w_fp32 = weight.detach().to(torch.float32)
    abs_max = w_fp32.abs().amax(dim=-1, keepdim=True)
    scale = (abs_max / qmax).clamp(min=1e-8)

    q = torch.round(w_fp32 / scale).clamp(qmin, qmax)
    dequant = q * scale
    return dequant.to(weight.dtype)


class Ablator:
    def __init__(self, model: nn.Module):
        self.model = model

    def _get_module(self, name: str) -> nn.Module:
        module: nn.Module = self.model
        for part in name.split("."):
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    @contextmanager
    def ablate(self, layer_names: Iterable[str], bits: int = 4):
        saved: dict[str, torch.Tensor] = {}
        try:
            for name in layer_names:
                module = self._get_module(name)
                if not isinstance(module, nn.Linear):
                    raise TypeError(
                        f"Layer '{name}' is {type(module).__name__}, "
                        "expected nn.Linear"
                    )
                saved[name] = module.weight.data.detach().to("cpu", copy=True)
                module.weight.data = fake_quantize(module.weight.data, bits)
            yield
        finally:
            for name, original in saved.items():
                module = self._get_module(name)
                target_device = module.weight.data.device
                module.weight.data = original.to(target_device, copy=True)


def list_linear_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear)
    ]


def is_lm_head(name: str) -> bool:
    return name in {"lm_head", "embed_out"} or name.endswith(".lm_head")
