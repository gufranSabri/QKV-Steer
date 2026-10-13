from __future__ import annotations

import torch
import torch.nn as nn

from qkv_steer._vendor.qkv_hooks import get_projection


def segment_bounds(m: int, n_segments: int, d: int) -> tuple[int, int]:
    if d % n_segments != 0:
        raise ValueError(f"feature dim D={d} is not divisible by n_segments={n_segments}")
    s = d // n_segments
    return m * s, (m + 1) * s


def make_write_hook(push_by_segment: dict[int, torch.Tensor | float], n_segments: int):
    def hook(module, inputs, output):
        d = output.shape[-1]
        modified = output.clone()
        for m, push in push_by_segment.items():
            lo, hi = segment_bounds(m, n_segments, d)
            push_t = torch.as_tensor(push, dtype=output.dtype, device=output.device)
            modified[..., lo:hi] = modified[..., lo:hi] + push_t
        return modified
    return hook


def register_steer_hooks(
    model, pushes: dict[tuple[int, str], dict[int, torch.Tensor]], n_segments: int
) -> list:
    handles = []
    for (layer_idx, view), push_by_segment in pushes.items():
        proj = get_projection(model, layer_idx, view)
        handle = proj.register_forward_hook(make_write_hook(push_by_segment, n_segments))
        handles.append(handle)
    return handles


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()
