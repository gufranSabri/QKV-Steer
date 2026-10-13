from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def _key_hash(key: dict) -> str:
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:16]


def load_or_compute_direction(
    cache_dir: str | Path, key: dict, compute_fn
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = Path(cache_dir) / _key_hash(key)
    direction_path = cache_dir / "direction.npy"
    sigma_path = cache_dir / "sigma.npy"

    if direction_path.exists() and sigma_path.exists():
        direction = torch.from_numpy(np.load(direction_path))
        sigma = torch.from_numpy(np.load(sigma_path))
        return direction, sigma

    direction, sigma = compute_fn()
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(direction_path, direction.numpy())
    np.save(sigma_path, sigma.numpy())
    return direction, sigma
