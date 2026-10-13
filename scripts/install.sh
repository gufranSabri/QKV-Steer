#!/bin/bash
# QKV-Steer dependency install -- the single source of truth, there is no
# requirements.txt. Every SLURM script sources this.
# Assumes the venv is already created and activated.
#
#   bash scripts/install.sh               # core deps only
#   bash scripts/install.sh --bleurt      # + BLEURT evaluation
#   bash scripts/install.sh --llm-judge   # + anthropic SDK for the LLM-judge evaluation

set -euo pipefail

WITH_BLEURT=0
WITH_LLM_JUDGE=0
for arg in "$@"; do
  [ "$arg" = "--bleurt" ] && WITH_BLEURT=1
  [ "$arg" = "--llm-judge" ] && WITH_LLM_JUDGE=1
done

echo "[install] core dependencies"

PIP_FLAGS=""
if [ -n "${CC_CLUSTER:-}${SLURM_JOB_ID:-}" ] && pip install --no-index --dry-run pip >/dev/null 2>&1; then
  PIP_FLAGS="--no-index"   # Compute Canada: install from the local wheelhouse
fi

pip install $PIP_FLAGS --upgrade pip

pip install $PIP_FLAGS \
  torch \
  transformers \
  datasets \
  numpy \
  scikit-learn \
  pyyaml \
  tqdm \
  accelerate \
  pytest

echo "[install] core dependencies OK"

if [ "$WITH_LLM_JUDGE" -eq 1 ]; then
  echo "[install] anthropic SDK (LLM-judge evaluation)"
  pip install $PIP_FLAGS anthropic
fi

# Git install, so it can NOT come from the offline wheelhouse -- these lines
# deliberately drop $PIP_FLAGS and hit the network.
if [ "$WITH_BLEURT" -eq 1 ]; then
  echo "[install] BLEURT (TensorFlow-CPU)"

  if [ ! -d bleurt ]; then
    git clone --depth 1 https://github.com/google-research/bleurt.git
  fi
  pip install ./bleurt

  mkdir -p models
  if [ ! -d models/BLEURT-20-D12 ]; then
    echo "[install] downloading BLEURT-20-D12 checkpoint"
    wget -q https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
    unzip -q BLEURT-20-D12.zip -d models/
    rm -f BLEURT-20-D12.zip
  fi

  python - <<'PY'
from bleurt import score
s = score.BleurtScorer("models/BLEURT-20-D12")
hi, lo = s.score(references=["Paris", "Paris"], candidates=["Paris", "a fish"])
assert hi > lo, "BLEURT is loaded but scoring nonsense"
print(f"[install] BLEURT OK  (match={hi:.3f}  mismatch={lo:.3f})")
PY
fi

echo "[install] done"
