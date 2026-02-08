# LOB Predictorium — GRU Baseline

## Objective

Optimise **Weighted Pearson correlation** for short-horizon price movement prediction on anonymised limit order book data.

The metric rewards:
- correct **direction**
- correct **relative magnitude**
- penalises overconfident scale errors

RMSE-style optimisation does **not** align well with leaderboard performance.

---

## Data

- 10,721 independent sequences
- 1000 timesteps per sequence
- Warm-up: steps 0–98
- Scored: steps 99–999

Each sequence is evaluated independently at inference time.

---

## Features

- 32 engineered LOB + trade features
- Roughly standardised and clipped
- Slight per-sequence scale drift

Per-sequence normalisation is beneficial.

---

## Targets

Two regression targets:

- **t0**: short-horizon price movement  
  Easier, strong microstructure signal

- **t1**: longer-horizon movement  
  Noisier, regime-dependent, main bottleneck

Improving t1 without harming t0 is key.

---

## Model

**GRU (Seq2Seq, streaming inference)**

- Hidden size: **128**
- Layers: **4**
- Dropout: **0.03**
- Stateful inference per sequence
- Full-sequence training (Seq2Seq)

This setup consistently outperforms windowed and shallow models.

---

## Loss Function (Key Insight)

Training with MSE or weighted MSE leads to:
- good validation loss
- worse leaderboard correlation

**Solution:** train directly on the competition metric.

- Loss = **negative Pearson correlation**
- Computed only on `need_prediction == 1` timesteps
- Strongly improves alignment between validation and leaderboard scores

---

## Training Details

- Optimiser: Adam
- Learning rate: 3e-4 – 1e-3
- Batch size: 32
- Early stopping on validation correlation
- Gradient clipping for stability

---

## Results

- Validation correlation ≈ **0.27**
- Significantly above provided baseline
- Validation and leaderboard scores closely aligned

---

## Key Observations

- 4-layer GRUs outperform shallow models
- Hidden size sweet spot: **64–128**
- Small dropout (≈0.03–0.06) improves generalisation
- Overfitting appears quickly when optimising correlation directly
- Conservative predictions perform better than aggressive ones

---

## Next Steps

- Light hyperparameter tuning around best config
- Multi-seed ensembling (if allowed)
- Further stabilise t1 predictions
- Explore regime-aware conditioning

This model forms a strong, simple, and reliable baseline.