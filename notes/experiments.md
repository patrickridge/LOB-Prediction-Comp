## Experiments

### Reference: Baseline GRU (provided ONNX)
Mean Weighted Pearson: 0.2595
- t0: 0.3884
- t1: 0.1306

Notes:
- t1 is consistently weaker than t0.
- Any preprocessing/post-processing must match training, otherwise scores collapse.

---

## 2026-01-16 — Input warm-up normalisation (inference-only) + clipping

**Change**
- Per-sequence z-score normalisation of the 100-step input window using warm-up steps 0–98.
- Output clipping to [-6, 6].

**Why**
- Reduce per-sequence distribution shift.

**Result**
Mean Weighted Pearson: 0.1512
- t0: 0.2307
- t1: 0.0718

**Interpretation**
- The ONNX GRU was trained on unnormalised inputs. Normalising at inference introduces distribution mismatch.

**Decision**
- Revert. Only apply input normalisation if we retrain with the same preprocessing.

---

## 2026-01-16 — Baseline restored (control)

**Change**
- Reverted inference-time input normalisation.
- Kept explicit output clipping.

**Result**
Mean Weighted Pearson: 0.2595
- t0: 0.3884
- t1: 0.1306

**Interpretation**
- Confirms the earlier drop was caused by input distribution mismatch, not environment/code.

---

## 2026-01-16 — EMA smoothing (post-hoc)

**Change**
- EMA smoothing on outputs, α = 0.2, reset per sequence.

**Why**
- Reduce noisy outputs, improve correlation.

**Result**
Mean Weighted Pearson: 0.1474
- t0: 0.2308
- t1: 0.0640

**Interpretation**
- EMA adds lag; hurts short-horizon alignment.
- Baseline GRU already smooths internally.

**Decision**
- Discard EMA smoothing.

---

## 2026-01-16 — Ridge baseline (lagged-window, quick sanity)

**Setup**
- Ridge regression on flattened lag window (K=20) using N=800 sequences for training.
- Local sanity metric: *unweighted* Pearson on valid (quick check).

**Result (unweighted Pearson, sanity)**
- t0: 0.2772
- t1: 0.0709
- mean: 0.1741

**Interpretation**
- Linear model captures signal but underperforms GRU on the actual competition metric.
- Next improvement path is feature engineering + weighting objective towards |target|.

**Next**
- Add engineered LOB features (mid, spread, imbalance, deltas).
- Try weighted training (weights = |target|) and/or XGBoost.