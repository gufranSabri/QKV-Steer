# Dispatch to a labeling scheme, so relabeling never requires re-extraction.

from __future__ import annotations

from qkv_steer.config import Config
from qkv_steer._vendor.bleurt import score_bleurt_batch
from qkv_steer._vendor.exact_match import score_exact_match


def label_examples(
    cfg: Config,
    answers: list[str],
    golds: list,
) -> list[tuple[float, int]]:
    # Returns [(score, label), ...]; label == 1 means HALLUCINATED.
    scheme = cfg.labeling.scheme

    if scheme == "exact_match":
        return [
            score_exact_match(cfg.dataset.name, ans, gold)
            for ans, gold in zip(answers, golds)
        ]

    if scheme == "bleurt":
        return score_bleurt_batch(
            answers,
            golds,
            checkpoint=cfg.labeling.bleurt_checkpoint,
            threshold=cfg.labeling.bleurt_threshold,
        )

    raise ValueError(f"unknown labeling scheme {scheme!r}")
