from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize_run(run_dir: Path, method: str | None) -> dict | None:
    metrics_path = run_dir / "steer_metrics.json"
    jsonl_path = run_dir / "steer_responses.jsonl"
    if not metrics_path.exists() or not jsonl_path.exists():
        return None

    all_metrics = json.loads(metrics_path.read_text())
    run_method = method if method in all_metrics else next(iter(all_metrics), None)
    if run_method is None:
        return None
    metrics = all_metrics[run_method]

    rows = [json.loads(line) for line in open(jsonl_path)]
    u_key = f"hallucination_label_unsteered_{run_method}"
    s_key = f"hallucination_label_steered_{run_method}"
    n_worse = sum(1 for r in rows if u_key in r and s_key in r and r[u_key] == 0 and r[s_key] == 1)
    n_better = sum(1 for r in rows if u_key in r and s_key in r and r[u_key] == 1 and r[s_key] == 0)

    return {
        "run_name": run_dir.name,
        "method": run_method,
        "hallucination_rate_unsteered": metrics["hallucination_rate_unsteered"],
        "hallucination_rate_steered": metrics["hallucination_rate_steered"],
        "hallucination_gt_truth": n_worse,
        "truth_gt_hallucination": n_better,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--method", default=None, help="eval method to summarize; defaults to whichever is present")
    parser.add_argument("--out", default="runs_summary.csv")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    summaries = []
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        summary = summarize_run(run_dir, args.method)
        if summary is not None:
            summaries.append(summary)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_name",
            "method",
            "hallucination_rate_unsteered",
            "hallucination_rate_steered",
            "hallucination_gt_truth",
            "truth_gt_hallucination",
        ])
        writer.writeheader()
        writer.writerows(summaries)

    print(f"wrote {len(summaries)} rows to {args.out}")


if __name__ == "__main__":
    main()
