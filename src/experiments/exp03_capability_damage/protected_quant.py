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


@contextmanager
def protected_quantization(
    model: nn.Module,
    sensitivity_csv: Path | str,
    *,
    bits: int = 4,
    protect_top_pct: float = 0.10,
    include_lm_head: bool = False,
) -> Iterator[None]:
    df = pd.read_csv(sensitivity_csv)
    df = df.dropna(subset=["kl_div"]).sort_values("kl_div", ascending=False)
    n_protect = max(int(len(df) * protect_top_pct), 1)
    protected_set = set(df.head(n_protect)["layer_name"].tolist())

    all_lm_layers = [n for n, _ in list_lm_linear_layers(model)]
    if not include_lm_head:
        all_lm_layers = [n for n in all_lm_layers if not is_lm_head(n)]

    quantize_layers = [n for n in all_lm_layers if n not in protected_set]
    log.info("Protected quant: %d layers @ FP16, %d layers @ INT%d",
             len(protected_set & set(all_lm_layers)), len(quantize_layers), bits)

    ablator = Ablator(model)
    with ablator.ablate(quantize_layers, bits=bits):
        yield
