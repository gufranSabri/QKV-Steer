#!/bin/bash
# Re-runs the winning SteerConfig (set via WINNING_CONFIG) on TriviaQA and CoQA.
# generate.py resumes per-example on its own (done_idx); RERUN=1 deletes a
# run's output first to force a clean regeneration from scratch.
set -euo pipefail

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

RERUN=${RERUN:-0}
LLM_ALIAS=${LLM_ALIAS:-llama3.1_8b}
WINNING_CONFIG=${WINNING_CONFIG:-configs/continuous_attribution_sign.yaml}
KEEP_CHANNELS=${KEEP_CHANNELS:-QK}
EXTRA_ARGS=${EXTRA_ARGS:-}
read -ra EVAL_METHODS <<< "${EVAL_METHODS:-lev}"

for DATASET in triviaqa coqa; do
  RUN_NAME="$(basename "$WINNING_CONFIG" .yaml)_${DATASET}_${KEEP_CHANNELS}"
  DEST="runs/${RUN_NAME}/steer_responses.jsonl"
  DETECTOR_CHECKPOINT="detectors/${LLM_ALIAS}_${DATASET}_${KEEP_CHANNELS}/best.pt"

  if [[ "$RERUN" == "1" && -f "$DEST" ]]; then
    rm -f "$DEST"
  fi

  echo "========================================"
  echo "Run: $RUN_NAME"
  echo "========================================"
  python3 -m qkv_steer.generate \
    --config "$WINNING_CONFIG" \
    --dataset "$DATASET" \
    --llm-alias "$LLM_ALIAS" \
    --run-name "$RUN_NAME" \
    --detector-checkpoint "$DETECTOR_CHECKPOINT" \
    --keep-channels "$KEEP_CHANNELS" \
    $EXTRA_ARGS

  if [[ ! -s "$DEST" ]]; then
    echo "  [error] $RUN_NAME produced no rows in $DEST -- skipping evaluation"
    continue
  fi

  for EVAL_METHOD in "${EVAL_METHODS[@]}"; do
    python3 -m qkv_steer.evaluate --run-name "$RUN_NAME" --method "$EVAL_METHOD"
  done
done
