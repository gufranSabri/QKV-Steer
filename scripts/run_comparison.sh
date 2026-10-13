#!/bin/bash
# Runs both SteerConfigs from the proposal (binary+mass-mean,
# continuous+attribution-sign) on every dataset in DATASETS, then evaluates each.
# generate.py resumes per-example on its own (done_idx); RERUN=1 deletes a
# run's output first to force a clean regeneration from scratch.
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RERUN=${RERUN:-0}
DATASETS=${DATASETS:-"truthfulqa"}
LLM_ALIAS=${LLM_ALIAS:-qwen2.5_7b}


for DATASET in $DATASETS; do
  DETECTOR_CHECKPOINT="detectors/${LLM_ALIAS}_${DATASET}/best.pt"

  for CONFIG in configs/binary_mass_mean.yaml configs/continuous_attribution_sign.yaml; do
    RUN_NAME="$(basename "$CONFIG" .yaml)_${DATASET}"
    DEST="runs/${RUN_NAME}/steer_responses.jsonl"

    if [[ "$RERUN" == "1" && -f "$DEST" ]]; then
      rm -f "$DEST"
    fi

    echo "========================================"
    echo "Run: $RUN_NAME"
    echo "========================================"

    python3 -m qkv_steer.generate \
      --config "$CONFIG" \
      --dataset "$DATASET" \
      --llm-alias "$LLM_ALIAS" \
      --run-name "$RUN_NAME" \
      --detector-checkpoint "$DETECTOR_CHECKPOINT" \
      --n-samples 10 --verbose
    

    if [[ ! -s "$DEST" ]]; then
      echo "  [error] $RUN_NAME produced no rows in $DEST -- skipping evaluation"
      continue
    fi

    python3 -m qkv_steer.evaluate --run-name "$RUN_NAME" --method bleurt
  done
done
