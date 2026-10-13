from __future__ import annotations

import torch

from qkv_steer._vendor.dataset import QKVFieldDataset


def binary_mask(saliency: torch.Tensor, threshold: float) -> torch.Tensor:
    avg = saliency.mean(dim=0)
    peak = avg.abs().max()
    if peak == 0:
        return torch.zeros_like(avg, dtype=torch.bool)
    return (avg.abs() / peak) >= threshold


def continuous_mask(saliency: torch.Tensor) -> torch.Tensor:
    avg = saliency.mean(dim=0)
    peak = avg.abs().max()
    if peak == 0:
        return avg
    return avg / peak


def mass_mean_direction(dataset: QKVFieldDataset) -> tuple[torch.Tensor, torch.Tensor]:
    sum_true = sum_false = None
    sq_true = sq_false = None
    n_true = n_false = 0

    for i in range(len(dataset)):
        field = dataset._load_raw(i)
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
