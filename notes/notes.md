## LOB Predictorium — Notes

source competition_package/env/bin/activate

### Objective
Optimise weighted Pearson correlation, not RMSE.
Metric rewards direction and relative magnitude.

Prefer:
- smooth
- conservative
- directionally correct predictions

### Data
- 10,721 independent sequences
- 1000 steps each
- Warm-up: 0–98
- Scored: 99–999

### Features
- 32 engineered LOB + trade features
- Roughly standardised with clipping around ±5
- Per-sequence normalisation likely beneficial

### Targets
- Heavy-tailed
- Very few samples exceed |6|
- Predictions clipped during scoring

### Targets intuition

- t0: short-horizon, directional price movement
  - Easier
  - Responds well to raw LOB imbalance

- t1: longer-horizon / noisier movement
  - Harder
  - Benefits from aggregation and regime-style features

Strategy:
- Use simple models + engineered features
- Focus improvements on t1 while preserving t0

### Modelling implications
- Use warm-up statistics
- Short-context models
- Avoid overconfident magnitude predictions

### Bid–ask spread inspection

Plotted p6 − p0 over a single sequence.

Key observations:
- Spread is small, noisy, and mean-reverting.
- Short-lived spikes decay quickly; no long-term drift.
- No visible change at the warm-up boundary (step 99).
- Negative values possible due to feature anonymisation.

Implications:
- Market regime is stable and liquid.
- Short-horizon microstructure signals dominate.
- Warm-up normalisation per sequence is appropriate.
- Conservative, short-context models are preferred.

### Next Plan
	1.	Train XGBoost / LightGBM
	•	Weight samples by |target|
	•	Use simple engineered LOB features
	•	Train on valid_small first
	2.	Evaluate on full validation set
	3.	If tree model ≥ GRU baseline
	•	Focus on better features
	•	Tune hyperparameters
	4.	Only retrain GRU if trees stop improving

### Current focus (Jan 2026)

- Build XGBoost baseline with engineered LOB features
- Train using sample weights = |target|
- Iterate on valid_small for fast feedback
- Promote to full valid only after feature set stabilises

Decision rule:
- If XGBoost ≥ GRU baseline (0.2595), continue with trees
- Only retrain GRU if tree-based models plateau

## Experiment 2026-01-17 — XGBoost v0 (K=20, unweighted)

**Change**
- Trained two XGBoost regressors (t0, t1) on flattened last-20-step window.

**Result (sanity check)**
- R² on valid_small (unweighted): t0 -0.056, t1 -0.099 (not a meaningful metric for this comp)

**Next**
- Score with ScorerStepByStep (Weighted Pearson) via solution_xgb.py.
- Retrain with sample weights: w = |target| (optionally clipped).

- Consider shifting focus to stateful GRU/LSTM + augmentation (per winner report): biggest gains came from variance-normalised augmentation + light noise, not heavy feature engineering.

## Submitting
zip -r submission.zip competition_package/example_solution/solution.py artifacts/gru_best_h128_L6_do0.1.pt

## Experiment 2026-01-30 — GRU (4-layer, hidden=32) — Seq2Seq

Model:
- GRU, 4 layers
- Hidden size: 32
- Dropout: 0.1
- Seq2Seq training (full sequence)
- Streaming stateful inference

Training:
- Batch size: 32
- Epochs: 20 (early stopping at 17)
- Optimizer: Adam (lr=1e-3)
- Device: Tesla T4 (Kaggle)

Validation (local):
- Weighted Pearson: ~0.283

Leaderboard:
- Weighted Pearson: **0.2699**


Keep the same model (h32/L4/do0.1) and try one of these high signal changes:
	•	LR = 3e-4 (everything else same)
	•	PATIENCE = 6 (since improvements are late, don’t cut it off early)
	•	EPOCHS = 40 (with patience 6, it will stop itself)

1) Same model, slightly lower LR (often improves generalisation)
	•	HIDDEN=32, NUM_LAYERS=4, DROPOUT=0.1
	•	LR = 7e-4
	•	EPOCHS = 60
	•	PATIENCE = 12

2) Same model, slightly less dropout (can help if underfitting)
	•	HIDDEN=32, NUM_LAYERS=4
	•	DROPOUT = 0.05
	•	keep LR=1e-3, EPOCHS=40, PATIENCE=10

3) Small capacity bump without blowing up overfitting
	•	HIDDEN = 48, NUM_LAYERS=4, DROPOUT=0.1
	•	keep LR=1e-3, EPOCHS=40, PATIENCE=10

Rule of thumb from your runs: H=32 + L=4 is the sweet spot, so tune LR/dropout/epochs around it.