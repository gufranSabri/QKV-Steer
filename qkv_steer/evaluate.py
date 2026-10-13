from __future__ import annotations

import argparse
import json
from pathlib import Path

from qkv_steer.evaluation import bleurt, llm_judge


def run(run_name: str, method: str, bleurt_checkpoint: str, bleurt_threshold: float):
    run_dir = Path("runs") / run_name
    rows = [json.loads(line) for line in open(run_dir / "steer_responses.jsonl")]

    prompts = [r["prompt"] for r in rows]
    golds = [r["gold"] for r in rows]
    unsteered = [r["response_unsteered"] for r in rows]
    steered = [r["response_steered"] for r in rows]

    if method == "bleurt":
        unsteered_scores = bleurt(unsteered, golds, bleurt_checkpoint, bleurt_threshold)
        steered_scores = bleurt(steered, golds, bleurt_checkpoint, bleurt_threshold)
        unsteered_labels = [label for _, label in unsteered_scores]
        steered_labels = [label for _, label in steered_scores]
    else:
        unsteered_labels = llm_judge(prompts, unsteered, golds)
        steered_labels = llm_judge(prompts, steered, golds)

    n = len(rows)
    n_diverged = sum(1 for r in rows if r.get("response_unsteered_diverged"))
    metrics = {
        "method": method,
        "n": n,
        "n_diverged_from_saved": n_diverged,
        "hallucination_rate_unsteered": sum(unsteered_labels) / n,
        "hallucination_rate_steered": sum(steered_labels) / n,
    }

    with open(run_dir / "steer_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--method", choices=["bleurt", "llm_judge"], default="bleurt")
    parser.add_argument("--bleurt-checkpoint", default="models/BLEURT-20-D12")
    parser.add_argument("--bleurt-threshold", type=float, default=0.5)
    args = parser.parse_args()
    run(args.run_name, args.method, args.bleurt_checkpoint, args.bleurt_threshold)


if __name__ == "__main__":
    main()
