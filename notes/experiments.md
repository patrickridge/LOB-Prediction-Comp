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

## Experiment 2026-01-17 — XGBoost (feature-based, online) baseline v1

**Setup**
- Train: `competition_package/datasets/train.parquet` filtered to `need_prediction == True`
- Valid small: `competition_package/datasets/valid_small.parquet` (for quick iteration)
- Full eval: `competition_package/datasets/valid.parquet` via `ScorerStepByStep` (Weighted Pearson)
- Model: two separate XGBoost regressors (t0 and t1)
- Features: base 32 + within-seq diffs for first 32 + rolling mean/std (win=10) for first 16
- Weights: `sample_weight = |target| + 1e-3`
- Artifacts: `artifacts/xgb_t0.json`, `artifacts/xgb_t1.json`

**Result (full valid, Weighted Pearson)**
- Mean: **0.216814**
- t0: **0.350630**
- t1: **0.082998**

**Observations**
- t0 is strong, t1 is weak → improvements should target t1 specifically.
- Feature pipeline + online inference wiring confirmed working end-to-end.

**Next**
- Add a small set of microstructure-style aggregate features (depth sums, imbalance, spread proxies).
- Tune XGB hyperparams separately for t1.
- Iterate on `valid_small`, only run full `valid` when a change looks promising.

## Experiment 2026-01-17 — XGBoost (engineered features) wired into submission-style predictor

**Train setup**
- Model: XGBoost regressors (separate models for t0 and t1)
- Train data: `train.parquet` rows where `need_prediction=True`
- Validation for training loop: `valid_small.parquet` (`need_prediction=True`)
- Weights: sample_weight = |target| + 1e-3 (aligns with weighted Pearson scoring emphasis)
- Features:
  - Base 32 LOB/trade features (p*, v*, dp*, dv*)
  - Diffs: `d_{c}` for first 32 base features (within-sequence)
  - Rolling stats (win=10): mean/std for first 16 base features (within-sequence)
  - LOB aggregates: `imbalance` and `spread`, plus rolling means `m10_imbalance`, `m10_spread`

**Fix**
- Updated `solution_xgb.py` to recompute `imbalance/spread` online per step and maintain rolling buffers.
- Ensured DMatrix includes `feature_names` to match trained model schema.

**Result (official scorer on full valid.parquet)**
- Mean Weighted Pearson: **0.217150**
  - t0: **0.351554**
  - t1: **0.082746**

**Takeaway**
- XGB + simple engineered features is below pretrained GRU baseline (0.2595), mainly due to weak t1.
- Pipeline is now correct → future gains should come from better features + objective/params, not wiring.

## Experiment 2026-01-18 — Train GRU (PyTorch) + streaming inference score

**Training setup**
- Model: GRU (hidden=128) + linear head, predicts [t0, t1] per step
- Data: full train.parquet + valid_small.parquet (sequence dataset)
- Loss: weighted MSE with weights = |target| (clamped)
- Epochs: 3, batch_size=8
- Saved: `artifacts/gru.pt`

**Training losses**
- epoch 1: train_loss=3.2520, valid_loss=28.2745
- epoch 2: train_loss=3.1949, valid_loss=28.0525
- epoch 3: train_loss=3.1731, valid_loss=27.5071

**Leaderboard-metric evaluation (ScorerStepByStep on valid.parquet)**
- Mean Weighted Pearson: **0.254709**
  - t0: 0.381497
  - t1: 0.127921

**Comparison**
- Pretrained ONNX GRU baseline: 0.259505 (t0 0.388378, t1 0.130631)
- Our GRU is close (-0.0048 absolute), suggests training + inference pipeline is mostly correct.

**Next**
- Improve training to close gap: match preprocessing, tune hidden size/lr/epochs, add input normalisation consistent in train + inference.
- Focus on boosting t1 (still the weak target).

## Experiment 2026-01-18 — GRU + warm-up norm + LayerNorm + weighted Huber

Changes:
- Per-sequence warm-up normalisation using steps 0–98
- Input LayerNorm before GRU
- Weighted Huber loss (δ = 1.0)

Results (valid):
- Mean Weighted Pearson: 0.2493
- t0: 0.3742
- t1: 0.1244

Interpretation:
- Performance slightly worse than baseline GRU (0.2547).
- Indicates baseline GRU already handled scale and noise well.
- Added normalisation + robust loss likely over-regularised the model.

Decision:
- Revert warm-up normalisation and Huber loss.
- Keep baseline GRU as reference.

## Experiment 2026-01-19 — GRU (2-layer, hidden=256)

Model:
- GRU, 2 layers
- Hidden size: 256
- Dropout: 0.1
- Streaming inference (stateful)
- Weighted MSE loss
- No warm-up normalisation

Training:
- Full sequences (T=1000)
- Batch size: 8
- Epochs: 3
- Device: MPS (Apple GPU)
- Total training time ~3.5 hours

Validation (local):
- Mean Weighted Pearson: 0.2602
- t0: 0.3871
- t1: 0.1332

