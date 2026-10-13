from __future__ import annotations

import torch


def continuous_mask(
    saliency: torch.Tensor, steer_direction: str = "both", clip_percentile: float | None = None
) -> torch.Tensor:
    avg = saliency.mean(dim=0)
    if clip_percentile is not None:
        clip_value = torch.quantile(avg.abs().flatten(), clip_percentile)
        avg = avg.clamp(min=-clip_value, max=clip_value)
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
