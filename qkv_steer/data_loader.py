from __future__ import annotations

import json
from pathlib import Path

from qkv_steer._vendor.datasets import Example, load_examples
from qkv_steer._vendor.dataset import QKVFieldDataset
from qkv_steer._vendor.run_extraction import parse_meta


class DataLoader:
    def __init__(self, cfg, dataset_name: str, llm_alias: str):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.llm_alias = llm_alias
        self.root = cfg.dataset_dir_for(dataset_name, llm_alias)

    def training_fields(self) -> QKVFieldDataset:
        return QKVFieldDataset(self.root, keep_channels=self.cfg.model.keep_channels)

    def eval_examples(self) -> list[Example]:
        return load_examples(self.cfg)

    def unsteered_response(self, idx: int) -> dict:
        manifest = self.root / "manifest.jsonl"
        with open(manifest) as f:
            for line in f:
                rec = json.loads(line)
                if rec["idx"] == idx:
                    meta = parse_meta(self.root / rec["dir"] / "meta.txt")
                    return {
                        "response": meta["response"],
                        "score": float(meta["score"]),
                        "label": int(meta["label"]),
                    }
        raise KeyError(f"no example with idx={idx} in {manifest}")
