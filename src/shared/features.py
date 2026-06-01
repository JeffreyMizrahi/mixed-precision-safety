from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import torch
from torch import nn

from shared.models import classify_layer, extract_layer_index
from shared.utils import progress_bar, redirect_logging_to_tqdm

log = logging.getLogger(__name__)


@dataclass
class LayerFeatures:
    layer_name: str
    role: str
    layer_index: int | None
    layer_index_normalized: float
    n_in: int
    n_out: int
    n_params: int
    weight_kurtosis: float
    weight_max_abs: float
    weight_mean_abs: float
    weight_outlier_ratio: float
    effective_rank: float
    condition_proxy: float

    def to_dict(self) -> dict:
        return asdict(self)


def extract_layer_features(
    name: str,
    module: nn.Linear,
    n_blocks: int,
) -> LayerFeatures:
    weight = module.weight.data
    layer_idx = extract_layer_index(name)
    layer_idx_norm = (
        layer_idx / max(n_blocks - 1, 1) if layer_idx is not None else 0.0
    )

    flat = weight.flatten().float()
    mean = flat.mean()
    std = flat.std()
    if std > 0:
        kurt = ((flat - mean).pow(4).mean() / std.pow(4)).item() - 3.0
    else:
        kurt = 0.0
    max_abs = weight.abs().max().item()
    mean_abs = weight.abs().mean().item()
    outlier_ratio = max_abs / mean_abs if mean_abs > 0 else 0.0

    eff_rank, cond_proxy = _spectral_features(weight)

    return LayerFeatures(
        layer_name=name,
        role=classify_layer(name),
        layer_index=layer_idx,
        layer_index_normalized=layer_idx_norm,
        n_in=weight.shape[1],
        n_out=weight.shape[0],
        n_params=weight.numel(),
        weight_kurtosis=float(kurt),
        weight_max_abs=max_abs,
        weight_mean_abs=mean_abs,
        weight_outlier_ratio=outlier_ratio,
        effective_rank=eff_rank,
        condition_proxy=cond_proxy,
    )


def extract_all_features(
    layers: list[tuple[str, nn.Linear]],
    n_blocks: int,
) -> list[LayerFeatures]:
    out = []
    with redirect_logging_to_tqdm():
        for name, module in progress_bar(layers, desc="extract features", total=len(layers)):
            try:
                out.append(extract_layer_features(name, module, n_blocks))
            except Exception as e:
                log.warning("Feature extraction failed for %s: %s", name, e)
    return out


def count_blocks(layers: list[tuple[str, nn.Linear]]) -> int:
    indices = {extract_layer_index(n) for n, _ in layers}
    indices.discard(None)
    return len(indices)


def _spectral_features(weight: torch.Tensor) -> tuple[float, float]:
    w = weight.detach().to(torch.float32).cpu()
    if w.shape[0] > w.shape[1]:
        w = w.T
    try:
        s = torch.linalg.svdvals(w)
    except Exception:
        return 0.0, 0.0
    s = s[s > 1e-10]
    if s.numel() == 0:
        return 0.0, 0.0
    p = s / s.sum()
    entropy = -(p * torch.log(p + 1e-12)).sum().item()
    eff_rank = float(torch.exp(torch.tensor(entropy)).item())
    s_sorted, _ = torch.sort(s, descending=True)
    s_max = s_sorted[0].item()
    idx_5pct = max(int(0.05 * s_sorted.numel()), 1)
    s_5pct = s_sorted[idx_5pct].item()
    cond_proxy = s_max / s_5pct if s_5pct > 0 else 0.0
    return eff_rank, cond_proxy
