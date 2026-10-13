from __future__ import annotations

import argparse
import json
from pathlib import Path

from qkv_steer.evaluation import bleurt, lev_judge, llm_judge_cloud
from qkv_steer.generate import write_csv


def _append_judgments_to_txt(txt_path: Path, method: str, rows: list[dict]) -> None:
    if not txt_path.exists():
        return
    text = txt_path.read_text()
    blocks = text.split("-------\n")
    out = []
    row_i = 0
    for block in blocks:
        if not block.strip():
            out.append(block)
            continue
        if row_i < len(rows):
            row = rows[row_i]
            block = block.rstrip("\n")
            block += (
                f"\n{method} unsteered: label={row[f'hallucination_label_unsteered_{method}']} "
                f"score={row[f'hallucination_score_unsteered_{method}']:.4f}"
                f"\n{method} steered: label={row[f'hallucination_label_steered_{method}']} "
                f"score={row[f'hallucination_score_steered_{method}']:.4f}\n"
            )
            row_i += 1
        out.append(block)
    txt_path.write_text("-------\n".join(out))


def run(run_name: str, method: str, bleurt_checkpoint: str, bleurt_threshold: float):
    run_dir = Path("runs") / run_name
    jsonl_path = run_dir / "steer_responses.jsonl"
    rows = [json.loads(line) for line in open(jsonl_path)]

    prompts = [r["prompt"] for r in rows]
    golds = [r["gold"] for r in rows]
    unsteered = [r["response_unsteered"] for r in rows]
    steered = [r["response_steered"] for r in rows]

    if method == "bleurt":
        unsteered_scores = bleurt(unsteered, golds, bleurt_checkpoint, bleurt_threshold)
        steered_scores = bleurt(steered, golds, bleurt_checkpoint, bleurt_threshold)
    elif method == "lev":
        unsteered_scores = lev_judge(prompts, unsteered, golds)
        steered_scores = lev_judge(prompts, steered, golds)
    elif method == "llm_judge_cloud":
        unsteered_scores = llm_judge_cloud(prompts, unsteered, golds)
        steered_scores = llm_judge_cloud(prompts, steered, golds)
    else:
        raise ValueError(f"unknown method {method!r}")

    for row, (u_score, u_label), (s_score, s_label) in zip(rows, unsteered_scores, steered_scores):
        row[f"hallucination_score_unsteered_{method}"] = u_score
        row[f"hallucination_label_unsteered_{method}"] = u_label
        row[f"hallucination_score_steered_{method}"] = s_score
        row[f"hallucination_label_steered_{method}"] = s_label

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    write_csv(jsonl_path, jsonl_path.with_suffix(".csv"))
    _append_judgments_to_txt(run_dir / "responses.txt", method, rows)

    unsteered_labels = [r[f"hallucination_label_unsteered_{method}"] for r in rows]
    steered_labels = [r[f"hallucination_label_steered_{method}"] for r in rows]

    n = len(rows)
    n_diverged = sum(1 for r in rows if r.get("response_unsteered_diverged"))
    p_deltas = [
        r["p_hallucinated_steered"] - r["p_hallucinated_unsteered"]
        for r in rows if "p_hallucinated_unsteered" in r
    ]
    metrics_path = run_dir / "steer_metrics.json"
    all_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    all_metrics[method] = {
        "n": n,
        "n_diverged_from_saved": n_diverged,
        "hallucination_rate_unsteered": sum(unsteered_labels) / n,
        "hallucination_rate_steered": sum(steered_labels) / n,
        "mean_detector_prob_delta": sum(p_deltas) / len(p_deltas) if p_deltas else None,
    }

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(json.dumps(all_metrics[method], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", choices=["bleurt", "lev", "llm_judge_cloud"], default="lev")
    parser.add_argument("--bleurt-checkpoint", default="models/BLEURT-20-D12")
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.run_name, args.method, args.bleurt_checkpoint, args.bleurt_threshold)


if __name__ == "__main__":
    main()
