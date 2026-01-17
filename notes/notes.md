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