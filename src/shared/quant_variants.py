from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
from torch import nn

from shared.ablator import Ablator, is_lm_head
from shared.models import list_lm_linear_layers

log = logging.getLogger(__name__)

VARIANTS = ("fp16", "int4_naive", "int4_protected")


@contextmanager
def quantized_variant(
    model: nn.Module,
    variant: str,
    *,
    sensitivity_csv: Path | str | None = None,
    bits: int = 4,
    protect_top_pct: float = 0.10,
) -> Iterator[None]:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")

    if variant == "fp16":
        log.info("Variant fp16: no quantization")
        yield
        return

    all_lm = [n for n, _ in list_lm_linear_layers(model) if not is_lm_head(n)]

    if variant == "int4_naive":
        log.info("Variant int4_naive: quantizing %d LM linear layers to INT%d",
                 len(all_lm), bits)
        with Ablator(model).ablate(all_lm, bits=bits):
            yield
        return

    if sensitivity_csv is None:
        raise ValueError("int4_protected requires sensitivity_csv from Exp 1")
    df = pd.read_csv(sensitivity_csv).dropna(subset=["kl_div"])
    df = df.sort_values("kl_div", ascending=False)
    n_protect = max(int(len(df) * protect_top_pct), 1)
    protected = set(df.head(n_protect)["layer_name"].tolist())
    to_quantize = [n for n in all_lm if n not in protected]
    log.info("Variant int4_protected: %d protected @ FP16, %d quantized @ INT%d",
             len(protected & set(all_lm)), len(to_quantize), bits)
    with Ablator(model).ablate(to_quantize, bits=bits):
        yield
