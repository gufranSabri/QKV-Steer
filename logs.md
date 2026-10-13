# Reading QKV-Steer's logs

`generate.py` writes two kinds of output per run, both under `runs/{run_name}/`:

- `steer_responses.jsonl` — always written, one line per example (the actual results).
- `steer_responses.csv` — the same rows as the jsonl, flattened to a spreadsheet-friendly table (`gold`'s list gets joined with `"; "`). Rewritten from the complete jsonl at the end of every run, so it's always in sync even if the run was resumed across multiple invocations.
- `generate.log` — only has real content when `--verbose` is on. Tensor-level diagnostics, written as plain text via Python's `logging`, so nothing prints to your terminal; `tail -f runs/{run_name}/generate.log` to watch it live.
- `figures/` — only populated when `--verbose` is on. One PNG + three `.npy` files per example, plus one summary PNG for the whole run.

Progress (`[run_name] idx=N done`) always prints to stdout/the SLURM log regardless of `--verbose`, so you always get a heartbeat.

## The fast path: steer_metrics.json

`evaluate.py` can be run multiple times against the same run with different `--method`s (`bleurt`, `lev`, `llm_judge_cloud`); each call merges its own results into `runs/{run_name}/steer_metrics.json` under its own key, rather than overwriting the file:

```json
{
  "lev": {
    "n": 817,
    "n_diverged_from_saved": 112,
    "hallucination_rate_unsteered": 0.41,
    "hallucination_rate_steered": 0.33,
    "mean_detector_prob_delta": -0.052
  },
  "llm_judge_cloud": {
    "n": 817,
    "n_diverged_from_saved": 112,
    "hallucination_rate_unsteered": 0.38,
    "hallucination_rate_steered": 0.29,
    "mean_detector_prob_delta": -0.052
  }
}
```

Three judges are available via `--method`:
- **`lev`** (default) — [Lev](https://huggingface.co/interfaze-ai/lev), a small local model (Qwen3.5-4B + LoRA) that answers a yes/no "does this hallucinate?" question directly as a calibrated probability, no generated text to parse, fully offline.
- **`llm_judge_cloud`** — Qwen3.8-27B via Novita AI's cloud API, prompted to output a single `0`/`1` character (needs `NOVITA_API_KEY` in `.env` at the repo root or exported in your shell — see `scripts/install.sh`).
- **`bleurt`** — lexical/semantic similarity to the gold answer(s) via BLEURT-20-D12, no LLM judging involved.

Fields, same for every method:
- **`hallucination_rate_unsteered` / `_steered`** — the fraction of responses that method's judge calls hallucinated, before and after steering. This is the proposal's actual success metric. Steering working means `_steered` is meaningfully lower than `_unsteered`.
- **`mean_detector_prob_delta`** — average change in the *detector's own* P(hallucinated), steered minus unsteered, across every example. Identical across every method's entry (it doesn't depend on the judge), since it's the detector's own opinion, not the judge's. Negative means steering is pushing the detector's own opinion down on average. This is a different (and faster-to-check) signal than the judge-based rate: the judge asks "did the actual words get more correct," the detector delta asks "did the thing we computed the steering direction from move the way we told it to." They can disagree — e.g. steering could fool the detector (negative delta) without actually improving factual correctness (unchanged hallucination rate), which is itself a meaningful finding.
- **`n_diverged_from_saved`** — how many examples' live-regenerated unsteered response didn't match what QKV-Lens originally saved for that example (see below). A handful is normal; see the next section if it's a large fraction of `n`.

If you only check one number, check `mean_detector_prob_delta` first (it's always computed, independent of which judge you ran) and `hallucination_rate_steered` vs `_unsteered` second. Comparing `lev` against `llm_judge_cloud` on the same run is itself informative — if a cheap local judge and a 27B cloud judge disagree a lot, that's worth knowing before trusting either one's number in isolation.

## Per-example rows in steer_responses.jsonl

```json
{
  "idx": 2,
  "prompt": "...",
  "gold": ["..."],
  "response_unsteered": "...",
  "response_unsteered_diverged": true,
  "response_steered": "...",
  "p_hallucinated_unsteered": 0.21,
  "p_hallucinated_steered": 0.21,
  "hallucination_label_unsteered_lev": 1,
  "hallucination_score_unsteered_lev": 0.87,
  "hallucination_label_steered_lev": 0,
  "hallucination_score_steered_lev": 0.12
}
```

- **`response_unsteered`** is what the *live* regeneration produced, and is what the saliency map, mask, and steering push were actually computed from. This is the one `evaluate.py` scores.
- **`response_unsteered_diverged`** is `true` when this differs from what QKV-Lens's own original extraction run produced for this exact prompt. This is *not* an error — it means floating-point numerics diverged between QKV-Lens's original batched extraction (`batch_size=8`, left-padded) and `generate.py`'s single-example regeneration (`batch_size=1`, unpadded). Padding changes the attention computation's numerics slightly; occasionally that's enough to flip an argmax at a close decision point, and once one token flips, everything downstream can read completely differently (see idx=3 in a real run: `' Scoville scale'` vs `' The placenta'` — same prompt, genuinely different greedy path). It's expected on a meaningful minority of examples, especially with Qwen. It is a problem only if `n_diverged_from_saved` is a large fraction of `n` — that would suggest something more systematic (wrong dtype, wrong prompt template, etc.), not just batching noise.
- **`p_hallucinated_unsteered` / `_steered`** are the per-example version of `mean_detector_prob_delta` above.
- **`hallucination_label_{unsteered,steered}_{method}`** / **`hallucination_score_{unsteered,steered}_{method}`** only appear after `evaluate.py --method {method}` has been run against this run at least once. Each method writes its own suffixed keys (`_lev`, `_bleurt`, `_llm_judge_cloud`), so running `evaluate.py` with multiple methods accumulates all of them side by side on the same row instead of the later call overwriting the earlier one's judgment.

## generate.log (only with --verbose)

One block per example, in this order:

```
idx=2 field.Q: shape=(26, 28, 32) norm=33.79 mean=0.01524 std=0.2209 min=-2.373 max=2.672
idx=2 field.K: shape=(26, 28, 32) norm=396 mean=-0.02145 std=2.594 min=-63.82 max=31.82
idx=2 field.V: shape=(26, 28, 32) norm=45.25 mean=0.001267 std=0.2964 min=-3.679 max=3.205
```
The raw (unsteered) QKV feature field for this response. `(T, L, M)` = (generated tokens, layers, segments). This is the baseline scale everything else should be judged against — K is consistently the widest-ranging view here (note `min=-63.82`), which is normal: Q/K/V have different magnitudes by construction.

```
idx=2 saliency.Q: shape=(26, 28, 32) norm=0.7447 mean=-5.361e-05 std=0.004879 min=-0.05596 max=0.08173
```
Per-token Integrated-Gradients attribution at each `(layer, segment)` cell, signed. Positive = this cell pushed the detector toward "hallucinated"; negative = pushed away. This is *before* averaging over tokens. If `std` here is tiny relative to what you'd expect (compare across examples), the detector barely reacted to this response at all, and don't expect the resulting mask to do anything useful.

```
idx=2 mask: 0/2688 cells selected (0.0%)
idx=2 mask.Q: shape=(28, 32) norm=0 mean=0 std=0 min=0 max=0
```
Binary mode: how many of the 2688 `(L×M×3)` cells cleared the relative-to-peak threshold (`threshold=0.5` means "at least half this response's own peak |attribution|"). **0% selected is common and not automatically a bug** — it means this response's attribution map had no cell standing out sharply above the rest (a flat/diffuse saliency map). It becomes worth investigating if it's 0% on *every* example in a run; that suggests the threshold is miscalibrated for this model/dataset, not that steering is broken.

Continuous mode instead prints the full `[-1, 1]`-normalized tensor stats per view, no selection count — every cell gets pushed a little, scaled by its own value.

```
idx=2 pushes: empty (no cells pushed)
```
or
```
idx=3 pushes: 1 (layer,view) pairs, 1 cells pushed
idx=3 pushes.values: shape=(1,) norm=5.844e-05 ...
```
The actual scalar values about to be added into the model's raw projection output. **This is the most important sanity check in the whole log.** Compare `pushes.values`' `norm`/`max` against `field`'s own `norm` a few lines up. If the push is many orders of magnitude smaller than the field itself (e.g. `5.8e-05` vs. a field norm of `33`), the push is numerically inert — it will not move the model's output in any way that survives, no matter how many cells got selected. This exact pattern is what caught an earlier threshold-calibration bug: cells were being "selected" but the resulting push was too small to matter, which is why `P(hallucinated)` showed `delta +0.0000` on every example.

```
idx=2 P(hallucinated): 0.2108 -> 0.2108 (delta +0.0000)
```
The actual causal readout for this example: the detector's own probability before and after steering. This is the line to scan down the whole log for. All-zero deltas across a run means steering had no effect (check the push magnitudes above); consistently negative deltas (for the un-flipped direction) is the result you want; **the sign-flip sanity check** (negate `alpha` or the direction and confirm deltas go positive instead) is what tells you whether this is real causal steering or coincidence — do this once before trusting a comparison between configs.

Once per run (binary mode only, logged before the per-example loop):
```
direction.Q: shape=(28, 32) norm=0.2768 mean=0.0005931 std=0.009233 min=-0.02909 max=0.04803
sigma.Q: shape=(28, 32) norm=2.213 mean=0.06714 std=0.03096 min=0.01374 max=0.4419
```
The cached mass-mean direction and per-cell standard deviation for this `(dataset, model)` pair, computed once from the training corpus and reused across every example in the run (and across future runs on the same dataset/model, since it's disk-cached).

## Figures (only with --verbose)

`runs/{run_name}/figures/idxNNNNN.png` — one 3x3 grid per example: rows are *saliency (averaged over tokens)*, *mask*, *push*; columns are Q/K/V. Each cell is an `(L, M)` heatmap, layer on the y-axis, segment on the x-axis, red/blue diverging color (red = positive, blue = negative, white = zero/not selected). The title bar shows the P(hallucinated) before/after for that example.

What to look for:
- **Saliency row**: where is attribution concentrated? A map that's mostly white/pale with a few sharp red or blue patches means the detector has a localized opinion about specific layers/segments. A uniformly speckled map with no structure means the detector's attribution is closer to noise for this response.
- **Mask row**: in binary mode, this should visually pick out the sharpest patches from the saliency row above it — if the mask row is blank (all white) while the saliency row clearly has colored patches, the threshold is too strict for this response. In continuous mode the mask row should look like a smoothed/rescaled copy of the saliency row.
- **Push row**: should be non-blank wherever the mask row is non-blank, and its color should be the *opposite* sign of the saliency row at the same cell (steering pushes against the direction attribution says is driving hallucination). If the push row looks blank while the mask row isn't, the push magnitude is too small to render/matter — same issue the log's `pushes.values` line would also catch.

Alongside each PNG: `idxNNNNN_saliency.npy`, `idxNNNNN_mask.npy`, `idxNNNNN_pushes.npy` — the raw `(L, M, 3)` arrays behind the plot, for your own analysis (e.g. `np.load(...)`, aggregate across examples, compare configs) without needing to re-run generation.

`runs/{run_name}/figures/summary_deltas.png` — a histogram of `P(hallucinated)` deltas across every example in the run (read back from `steer_responses.jsonl`, so it reflects the whole run even if you resumed it across multiple invocations). A histogram concentrated to the left of zero is the run-level version of "steering is working"; concentrated at exactly zero is the run-level version of "steering did nothing."
