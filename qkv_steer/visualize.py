from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _pushes_to_dense(pushes: dict, shape: tuple[int, int, int]) -> np.ndarray:
    dense = np.zeros(shape, dtype=np.float32)
    views = ("Q", "K", "V")
    for (l, view), push_by_segment in pushes.items():
        ci = views.index(view)
        for m, v in push_by_segment.items():
            dense[l, m, ci] = float(v.item()) if torch.is_tensor(v) else float(v)
    return dense


def save_example_figure(
    out_dir: Path,
    idx: int,
    saliency: torch.Tensor,
    mask: torch.Tensor,
    pushes: dict,
    p_before: float,
    p_after: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    views = ("Q", "K", "V")

    saliency_avg = saliency.mean(dim=0).detach().cpu().numpy()
    mask_np = mask.detach().cpu().float().numpy()
    push_np = _pushes_to_dense(pushes, mask_np.shape)

    np.save(out_dir / f"idx{idx:05d}_saliency.npy", saliency_avg)
    np.save(out_dir / f"idx{idx:05d}_mask.npy", mask_np)
    np.save(out_dir / f"idx{idx:05d}_pushes.npy", push_np)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    panels = [("saliency (avg over tokens)", saliency_avg), ("mask", mask_np), ("push", push_np)]

    for row, (title, data) in enumerate(panels):
        vmax = np.abs(data).max() or 1.0
        for col, view in enumerate(views):
            ax = axes[row, col]
            im = ax.imshow(data[:, :, col], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_title(f"{title}: {view}", fontsize=9)
            ax.set_xlabel("segment (M)")
            ax.set_ylabel("layer (L)")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"idx={idx}  P(hallucinated): {p_before:.4f} -> {p_after:.4f} "
        f"(delta {p_after - p_before:+.4f})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_dir / f"idx{idx:05d}.png", dpi=130)
    plt.close(fig)


def save_run_summary_figure(out_dir: Path, deltas: list[float]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not deltas:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(deltas, bins=20, color="#4C72B0", edgecolor="black")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("P(hallucinated) delta (steered - unsteered)")
    ax.set_ylabel("count")
    ax.set_title(f"Steering effect across {len(deltas)} examples")
    fig.tight_layout()
    fig.savefig(out_dir / "summary_deltas.png", dpi=130)
    plt.close(fig)
