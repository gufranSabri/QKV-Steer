#!/bin/bash
# QKV-Steer dependency install -- the single source of truth, there is no
# requirements.txt. Every SLURM script sources this.
# Assumes the venv is already created and activated.
#
#   bash scripts/install.sh               # core deps + Lev (default evaluator)
#   bash scripts/install.sh --bleurt      # + BLEURT evaluation

set -euo pipefail

WITH_BLEURT=0
for arg in "$@"; do
  [ "$arg" = "--bleurt" ] && WITH_BLEURT=1
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
  matplotlib \
  openai \
  pytest

echo "[install] core dependencies OK"

# Lev (interfaze-ai/lev): the default evaluate.py judge. Git install, so it
# can NOT come from the offline wheelhouse -- deliberately drops $PIP_FLAGS
# and hits the network, same as BLEURT below.
echo "[install] Lev (local hallucination judge)"
pip install "lev[serve] @ git+https://github.com/Abhinavexists/lev#subdirectory=packages/lev"

python - <<'PY'
import lev
model = lev.load("interfaze-ai/lev")
result = model.system_one(
    "The sky is blue during a clear day.",
    {"true_statement": {"type": "noul", "instructions": "Is this statement true?"}},
)
p = result.answers["true_statement"].noul
assert p > 0.5, f"Lev is loaded but scoring nonsense (p={p})"
print(f"[install] Lev OK  (p_true={p:.3f})")
PY

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
