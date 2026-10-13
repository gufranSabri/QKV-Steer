from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from qkv_steer._vendor.seed import pick_device, seed_everything
from qkv_steer.config import KEEP_CHANNELS_COMBOS, load_steer_config
from qkv_steer.data_loader import DataLoader
from qkv_steer.debug import log_field_by_view, log_mask, log_pushes
from qkv_steer.directions import continuous_mask
from qkv_steer.llm_wrapper import LLM_Wrapper
from qkv_steer.visualize import save_example_figure, save_run_summary_figure

logger = logging.getLogger("qkv_steer.generate")


def setup_file_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(handler)


def write_csv(jsonl_path: Path, csv_path: Path) -> None:
    rows = [json.loads(line) for line in open(jsonl_path)]
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            if isinstance(row.get("gold"), list):
                row["gold"] = "; ".join(str(g) for g in row["gold"])
            writer.writerow(row)


def build_push_continuous(mask, alpha: float, sign: float = -1.0) -> dict:
    pushes: dict = {}
    n_segments = mask.shape[1]
    for l in range(mask.shape[0]):
        for ci, view in enumerate(("Q", "K", "V")):
            push_by_segment = {m: sign * alpha * mask[l, m, ci].item() for m in range(n_segments)}
            pushes[(l, view)] = push_by_segment
    return pushes


def run(
    cfg, steer_cfg, dataset_name: str, llm_alias: str, run_name: str, detector_checkpoint: str,
    verbose: bool = False,
):
    seed_everything(cfg.train.seed)
    device = pick_device()

    out_dir = Path("runs") / run_name
    setup_file_logging(out_dir / "generate.log")

    loader = DataLoader(cfg, dataset_name, llm_alias)
    wrapper = LLM_Wrapper(cfg, device, detector_checkpoint=detector_checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "steer_responses.jsonl"
    figures_dir = out_dir / "figures"
    txt_path = out_dir / "responses.txt"

    done_idx = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                done_idx.add(json.loads(line)["idx"])

    txt_f = open(txt_path, "a") if verbose else None

    with open(out_path, "a") as out_f:
        for example in loader.eval_examples():
            if example.idx in done_idx:
                continue

            unsteered = loader.unsteered_response(example.idx)
            gen = wrapper.generate(example.prompt, cfg.dataset.max_new_tokens)
            diverged = gen["response"] != unsteered["response"]
            if diverged:
                logger.warning(
                    f"idx={example.idx}: live regeneration diverged from QKV-Lens's "
                    f"saved unsteered response (numeric drift from batch_size=1 vs the original "
                    f"extraction's batching) -- using the live one as the steering baseline.\n"
                    f"  saved: {unsteered['response']!r}\n"
                    f"  live:  {gen['response']!r}"
                )
            if verbose:
                log_field_by_view(f"idx={example.idx} field", gen["field"])

            p_before = wrapper.detect(gen["field"])
            hallucinated = p_before >= steer_cfg.hallucination_threshold

            if not hallucinated and steer_cfg.steer_nonhallucinations == "skip":
                steer_mode = "skip"
                saliency = mask = pushes = None
                steered = gen
            else:
                steer_mode = "up" if (not hallucinated and steer_cfg.steer_nonhallucinations == "up") else "down"
                saliency = wrapper.saliency(gen["field"])
                if verbose:
                    log_field_by_view(f"idx={example.idx} saliency", saliency)

                mask = continuous_mask(saliency, steer_cfg.steer_direction, steer_cfg.clip_percentile)
                pushes = build_push_continuous(mask, steer_cfg.alpha, sign=1.0 if steer_mode == "up" else -1.0)
                if verbose:
                    log_mask(f"idx={example.idx} mask", mask)
                    log_pushes(f"idx={example.idx} pushes", pushes)

                steered = wrapper.generate(
                    example.prompt, cfg.dataset.max_new_tokens, steer_pushes=pushes
                )

            p_after = p_before if steer_mode == "skip" else wrapper.detect(steered["field"])

            if verbose:
                log_field_by_view(f"idx={example.idx} steered_field", steered["field"])
                logger.info(
                    f"idx={example.idx} P(hallucinated): {p_before:.4f} -> {p_after:.4f} "
                    f"(delta {p_after - p_before:+.4f})"
                )
                if saliency is not None:
                    save_example_figure(figures_dir, example.idx, saliency, mask, pushes, p_before, p_after)

            row = {
                "idx": example.idx,
                "prompt": example.prompt,
                "gold": example.gold,
                "response_unsteered": gen["response"],
                "response_unsteered_diverged": diverged,
                "response_steered": steered["response"],
                "p_hallucinated_unsteered": p_before,
                "p_hallucinated_steered": p_after,
                "steer_mode": steer_mode,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

            if txt_f is not None:
                gold_text = "; ".join(str(g) for g in example.gold) if isinstance(example.gold, list) else str(example.gold)
                txt_f.write(
                    "-------\n"
                    f"prompt: {example.prompt}\n"
                    f"gold: {gold_text}\n"
                    f"unsteered: {gen['response']}\n"
                    f"steered: {steered['response']}\n"
                    "-------\n\n"
                )
                txt_f.flush()

            print(f"[{run_name}] idx={example.idx} done", flush=True)

    if txt_f is not None:
        txt_f.close()

    write_csv(out_path, out_path.with_suffix(".csv"))

    if verbose:
        all_deltas = []
        with open(out_path) as f:
            for line in f:
                row = json.loads(line)
                all_deltas.append(row["p_hallucinated_steered"] - row["p_hallucinated_unsteered"])
        if all_deltas:
            save_run_summary_figure(figures_dir, all_deltas)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--llm-alias", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector-checkpoint", required=True)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--steer-direction", choices=["both", "up_only", "down_only"], default=None)
    parser.add_argument("--keep-channels", choices=sorted(KEEP_CHANNELS_COMBOS), default=None)
    parser.add_argument("--steer-nonhallucinations", choices=["skip", "down", "up"], default=None)
    parser.add_argument("--hallucination-threshold", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    dataset_config = Path("configs/datasets") / args.dataset / f"{args.llm_alias}.yaml"
    overrides: dict = {}
    if args.n_samples is not None:
        overrides["dataset"] = {"n_samples": args.n_samples}
    if args.alpha is not None:
        overrides.setdefault("steer", {})["alpha"] = args.alpha
    if args.steer_direction is not None:
        overrides.setdefault("steer", {})["steer_direction"] = args.steer_direction
    if args.keep_channels is not None:
        overrides.setdefault("model", {})["keep_channels"] = KEEP_CHANNELS_COMBOS[args.keep_channels]
    if args.steer_nonhallucinations is not None:
        overrides.setdefault("steer", {})["steer_nonhallucinations"] = args.steer_nonhallucinations
    if args.hallucination_threshold is not None:
        overrides.setdefault("steer", {})["hallucination_threshold"] = args.hallucination_threshold
    cfg, steer_cfg = load_steer_config(args.config, dataset_config=dataset_config, overrides=overrides or None)
    run(
        cfg, steer_cfg, args.dataset, args.llm_alias, args.run_name, args.detector_checkpoint,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
