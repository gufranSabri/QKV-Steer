from __future__ import annotations

import torch

from qkv_steer._vendor.dataset import QKVFieldDataset, zero_channels


CLIP_PERCENTILE = 0.95


def binary_mask(saliency: torch.Tensor, threshold: float) -> torch.Tensor:
    avg = saliency.mean(dim=0).abs()
    clip_value = torch.quantile(avg.flatten(), CLIP_PERCENTILE)
    clipped = avg.clamp(max=clip_value)
    lo, hi = clipped.min(), clipped.max()
    if hi == lo:
        return torch.zeros_like(avg, dtype=torch.bool)
    normalized = (clipped - lo) / (hi - lo)
    return normalized >= threshold


def continuous_mask(saliency: torch.Tensor, steer_direction: str = "both") -> torch.Tensor:
    avg = saliency.mean(dim=0)
    peak = avg.abs().max()
    normalized = avg / peak if peak != 0 else avg

    if steer_direction == "up_only":
        # push = -alpha * mask, so a positive mask value pushes DOWN and a
        # negative value pushes UP -- zero the positive side to keep only up-steering.
        normalized = normalized.clamp(max=0)
    elif steer_direction == "down_only":
        normalized = normalized.clamp(min=0)
    elif steer_direction != "both":
        raise ValueError(f"steer_direction must be 'both', 'up_only', or 'down_only', got {steer_direction!r}")

    return normalized


def mass_mean_direction(
    dataset: QKVFieldDataset, keep_channels: list[str] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    sum_true = sum_false = None
    sq_true = sq_false = None
    n_true = n_false = 0

    for i in range(len(dataset)):
        field = dataset._load_raw(i)
        if keep_channels is not None:
            field = zero_channels(field, keep_channels)
        label = int(dataset.records[i]["label"])

        if sum_true is None:
            shape = field.shape[1:]
            sum_true = torch.zeros(shape, dtype=torch.float64)
            sum_false = torch.zeros(shape, dtype=torch.float64)
            sq_true = torch.zeros(shape, dtype=torch.float64)
            sq_false = torch.zeros(shape, dtype=torch.float64)

        tokens = field.double()
        if label == 1:
            sum_false += tokens.sum(dim=0)
            sq_false += (tokens ** 2).sum(dim=0)
            n_false += tokens.shape[0]
        else:
            sum_true += tokens.sum(dim=0)
            sq_true += (tokens ** 2).sum(dim=0)
            n_true += tokens.shape[0]

    if n_true == 0 or n_false == 0:
        raise ValueError(
            f"mass_mean_direction needs both labels present, got n_true={n_true} n_false={n_false}"
        )

    mean_true = sum_true / n_true
    mean_false = sum_false / n_false
    var_true = (sq_true / n_true) - mean_true ** 2
    var_false = (sq_false / n_false) - mean_false ** 2
    n_total = n_true + n_false
    pooled_var = (var_true.clamp(min=0) * n_true + var_false.clamp(min=0) * n_false) / n_total
    sigma = pooled_var.sqrt().clamp(min=1e-6)

    direction = (mean_true - mean_false).float()
    return direction, sigma.float()
