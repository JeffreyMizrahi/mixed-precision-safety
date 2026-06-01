from __future__ import annotations

import torch
import torch.nn.functional as F


def logit_kl(logits_p: torch.Tensor, logits_q: torch.Tensor, *, reduction: str = "mean") -> torch.Tensor:
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    if reduction == "none":
        return kl
    raise ValueError(f"Unknown reduction: {reduction}")


def safe_logit_kl(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[float, int]:
    p_f32 = logits_p.float()
    q_f32 = logits_q.float()

    pos_mask = attention_mask.bool() if attention_mask is not None else torch.ones(
        p_f32.shape[:-1], dtype=torch.bool, device=p_f32.device
    )
    finite_p = torch.isfinite(p_f32).all(dim=-1)
    finite_q = torch.isfinite(q_f32).all(dim=-1)
    pos_mask = pos_mask & finite_p & finite_q

    n_valid = int(pos_mask.sum().item())
    if n_valid == 0:
        return float("nan"), 0

    p_flat = p_f32[pos_mask]
    q_flat = q_f32[pos_mask]
    log_p = F.log_softmax(p_flat, dim=-1)
    log_q = F.log_softmax(q_flat, dim=-1)
    diff = (log_p - log_q).clamp(min=-50.0, max=50.0)
    p = log_p.exp()
    kl = (p * diff).sum(dim=-1)
    return float(kl.mean().item()), n_valid


def logit_jsd(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = m.clamp(min=1e-12).log()
    kl_pm = (p * (log_p - log_m)).sum(dim=-1)
    kl_qm = (q * (log_q - log_m)).sum(dim=-1)
    return (0.5 * (kl_pm + kl_qm)).mean()


def top1_disagreement(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    return (logits_p.argmax(-1) != logits_q.argmax(-1)).float().mean()


def safe_top1_disagreement(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> float:
    p_f32 = logits_p.float()
    q_f32 = logits_q.float()
    pos_mask = attention_mask.bool() if attention_mask is not None else torch.ones(
        p_f32.shape[:-1], dtype=torch.bool, device=p_f32.device
    )
    pos_mask = pos_mask & torch.isfinite(p_f32).all(dim=-1) & torch.isfinite(q_f32).all(dim=-1)
    if int(pos_mask.sum()) == 0:
        return float("nan")
    top_p = p_f32[pos_mask].argmax(dim=-1)
    top_q = q_f32[pos_mask].argmax(dim=-1)
    return float((top_p != top_q).float().mean().item())


def expected_calibration_error(probs: torch.Tensor, correct: torch.Tensor, n_bins: int = 15) -> float:
    probs = probs.float()
    correct = correct.float()
    bin_edges = torch.linspace(0, 1, n_bins + 1, device=probs.device)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (probs > lo) & (probs <= hi) if i > 0 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean().item()
        bin_conf = probs[mask].mean().item()
        ece += (mask.sum().item() / n) * abs(bin_acc - bin_conf)
    return ece
