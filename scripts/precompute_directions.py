from __future__ import annotations

import argparse
from pathlib import Path

from qkv_steer.cache import load_or_compute_direction
from qkv_steer.config import load_steer_config
from qkv_steer.data_loader import DataLoader
from qkv_steer.directions import mass_mean_direction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/steer_default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--llm-alias", required=True)
    parser.add_argument("--cache-dir", default="cache")
    args = parser.parse_args()

    dataset_config = Path("configs/datasets") / args.dataset / f"{args.llm_alias}.yaml"
    cfg, _ = load_steer_config(args.config, dataset_config=dataset_config)
    loader = DataLoader(cfg, args.dataset, args.llm_alias)

    n_segments = cfg.extract.n_segments
    if n_segments is None:
        from types import SimpleNamespace

        from transformers import AutoConfig

        from qkv_steer._vendor.qkv_hooks import read_geometry

        hf_config = AutoConfig.from_pretrained(cfg.llm.name)
        n_segments = read_geometry(SimpleNamespace(config=hf_config)).n_layers

    key = {
        "dataset": args.dataset,
        "llm_alias": args.llm_alias,
        "keep_channels": cfg.model.keep_channels,
        "n_segments": n_segments,
    }
    direction, sigma = load_or_compute_direction(
        args.cache_dir, key,
        lambda: mass_mean_direction(loader.training_fields(), keep_channels=cfg.model.keep_channels),
    )
    print(f"direction shape={tuple(direction.shape)} sigma shape={tuple(sigma.shape)}")


if __name__ == "__main__":
    main()
