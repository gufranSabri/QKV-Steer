from __future__ import annotations

import logging

import torch

logger = logging.getLogger("qkv_steer.verbose")


def stats_line(name: str, t: torch.Tensor) -> str:
    t = t.detach().float()
    if t.numel() == 0:
        return f"{name}: empty"
    return (
        f"{name}: shape={tuple(t.shape)} "
        f"norm={t.norm().item():.4g} mean={t.mean().item():.4g} std={t.std().item():.4g} "
        f"min={t.min().item():.4g} max={t.max().item():.4g}"
    )


def log_tensor(name: str, t: torch.Tensor) -> None:
    logger.info(stats_line(name, t))


def log_field_by_view(name: str, field: torch.Tensor) -> None:
    for ci, view in enumerate(("Q", "K", "V")):
        log_tensor(f"{name}.{view}", field[..., ci])


def log_mask(name: str, mask: torch.Tensor) -> None:
    log_tensor(name, mask)
    log_field_by_view(name, mask.float())


def log_pushes(name: str, pushes: dict) -> None:
    if not pushes:
        logger.info(f"{name}: empty (no cells pushed)")
        return
    all_vals = []
    for push_by_segment in pushes.values():
        for v in push_by_segment.values():
            all_vals.append(v.reshape(-1) if torch.is_tensor(v) else torch.tensor([float(v)]))
    flat = torch.cat(all_vals)
    n_layers_views = len(pushes)
    n_cells = sum(len(d) for d in pushes.values())
    logger.info(f"{name}: {n_layers_views} (layer,view) pairs, {n_cells} cells pushed")
    log_tensor(f"{name}.values", flat)
