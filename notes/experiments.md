---

# 📄 `experiments.md`

```md
# LOB Predictorium — Experiments

This file tracks experiments chronologically with decisions and outcomes.

---

## Reference Baseline

**Provided GRU (ONNX)**  
Mean Weighted Pearson: **0.2595**
- t0: 0.3884
- t1: 0.1306

Observation:
- t1 is consistently weaker
- Any preprocessing mismatch destroys performance

---

## Experiment 2026-01-17 — XGBoost v0

**Setup**
- Two regressors (t0, t1)
- Flattened last-20-step window
- No weighting

**Result**
- R² (unweighted, not meaningful): poor
- Weighted Pearson via scorer: < GRU baseline

**Conclusion**
- Trees underperform stateful GRU
- Abandoned for now

---

## Experiment 2026-01-30 — GRU Seq2Seq (32 / 4)

**Model**
- GRU
- Hidden = 32
- Layers = 4
- Dropout = 0.10
- Seq2Seq training
- Streaming inference

**Training**
- Adam, lr = 1e-3
- Early stopping at ~17 epochs

**Results**
- Local validation: ~0.283
- Leaderboard: **0.2699**

**Key Insight**
- Seq2Seq training is critical
- LB < val suggested loss mismatch

---

## Hyperparameter Sweep (Late Jan)

### Best configurations

| Hidden | Layers | Dropout | LB Score |
|------|--------|---------|----------|
| 128 | 4 | 0.03 | **0.2732** |
| 64 | 4 | 0.05 | 0.2728 |
| 32 | 4 | 0.10 | 0.2699 |

### Patterns
- 4 layers consistently beat 2
- Hidden sweet spot: **64–128**
- Small dropout (0.03–0.06) generalises best
- Too many layers with small hidden hurts

---

## Augmentation Experiments (Early Feb)

Tried:
- Variance normalisation
- Random scaling
- Gaussian noise

Result:
- Augmentation **did not improve** performance
- Often worsened validation correlation

Conclusion:
- Strong baseline already captures signal
- Augmentation unnecessary for this setup

---

## Critical Insight — Loss Function (Feb 2026)

From Giovanni:
> “I recommend using the metric itself as your loss function.”

### Change
- Replaced weighted MSE with **negative Pearson correlation**
- Implemented in PyTorch directly
- Masked on `need_prediction == 1`

### Effect
- Validation metric aligns with leaderboard
- Reduced gap between local and LB scores
- Training becomes more stable

---

## Experiment — Metric-Based Loss (128 / 4)

**Config**
- Hidden = 128
- Layers = 4
- Dropout = 0.03
- Loss = negative Pearson
- No augmentation

**Result**
- Best validation corr ≈ **0.2708**
- Much better calibration vs LB

---

## LR Tuning with Metric Loss

**Config**
- LR = 3e-4
- WD = 1e-5

**Result**
- Best validation corr ≈ **0.2708**
- More stable training
- Less overfitting late

**Decision**
- Use this checkpoint for submission

---

## Current Best Model (Submission)

- GRU (Seq2Seq, streaming)
- Hidden = 128
- Layers = 4
- Dropout = 0.03
- LR = 3e-4
- Weight decay = 1e-5
- **Loss = negative Pearson correlation**
- Validation corr ≈ **0.271**
- Leaderboard expected ≈ validation

---

## Open Questions

- Can t1 be stabilised further?
- Would ensembling seeds help?
- Would shallow attention on top of GRU help?

Baseline is now strong; remaining gains likely marginal.

## 2026-02-08 — GRU ensemble boost (metric-loss)

**Model**
- GRU seq2seq, streaming inference
- H=128, L=4, dropout=0.03

**Training**
- Loss: metric-based (weighted Pearson correlation; optimise -corr)
- LR=3e-4, WD=1e-5
- No augmentation
- Seeds trained: 42, 999

**Submission**
- Ensemble: mean of predictions from seed 42 + seed 999 checkpoints

**Result**
- Leaderboard: **0.2852** (improved from 0.2830 single model)

**Takeaway**
- Ensemble gives a clean uplift → worth adding 1–3 more diverse seeds/configs.

