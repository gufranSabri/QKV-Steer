#!/bin/bash
# Runs continuous+attribution-sign on every dataset in DATASETS, sweeping
# ALPHAS x STEER_DIRECTIONS, then evaluates each.
# generate.py resumes per-example on its own (done_idx); RERUN=1 deletes a
# run's output first to force a clean regeneration from scratch.
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RERUN=${RERUN:-0}
DATASETS=${DATASETS:-"truthfulqa"}
LLM_ALIAS=${LLM_ALIAS:-qwen2.5_7b}
N_SAMPLES=${N_SAMPLES:-10}
CONFIG=${CONFIG:-configs/continuous_attribution_sign.yaml}
read -ra ALPHAS <<< "${ALPHAS:-5 7 9 11 13 15}"
read -ra STEER_DIRECTIONS <<< "${STEER_DIRECTIONS:-both up_only down_only}"
read -ra EVAL_METHODS <<< "${EVAL_METHODS:-lev llm_judge_cloud}"

for ALPHA in "${ALPHAS[@]}"; do
  for DATASET in $DATASETS; do
    DETECTOR_CHECKPOINT="detectors/${LLM_ALIAS}_${DATASET}/best.pt"

    for STEER_DIRECTION in "${STEER_DIRECTIONS[@]}"; do

      RUN_NAME="$(basename "$CONFIG" .yaml)_${DATASET}_alpha${ALPHA}_${STEER_DIRECTION}"
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
        --n-samples "$N_SAMPLES" --verbose --alpha "$ALPHA" --steer-direction "$STEER_DIRECTION"

      if [[ ! -s "$DEST" ]]; then
        echo "  [error] $RUN_NAME produced no rows in $DEST -- skipping evaluation"
        continue
      fi

      for EVAL_METHOD in "${EVAL_METHODS[@]}"; do
        python3 -m qkv_steer.evaluate --run-name "$RUN_NAME" --method "$EVAL_METHOD"
      done
    done
  done
done
