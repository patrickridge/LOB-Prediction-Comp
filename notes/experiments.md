## Experiments

### Reference: Baseline GRU (provided ONNX)
Mean Weighted Pearson: 0.2595
- t0: 0.3884
- t1: 0.1306

Notes:
- t1 is consistently weaker than t0.
- Any preprocessing/post-processing must match training, otherwise scores collapse.

---

## Experiment Summary — Jan 2026 (GRU seq2seq)

### Baseline
- GRU baseline provided by competition
- Score: ~0.265

---

### GRU (hidden=128, layers=6, dropout=0.1)
- Training: seq2seq on full sequences
- Batch size: 32
- Early stopping (patience=3)
- Result (leaderboard): **0.258**
- Notes:
  - Deeper model but likely over-regularised
  - Did not outperform baseline

---

### GRU (hidden=256, layers=2, dropout=0.05)
- Training: seq2seq
- Batch size: 32
- Early stopping
- Result (leaderboard): **0.258**
- Notes:
  - Larger hidden size alone not sufficient
  - Shallow depth likely limits representation

---

### GRU (hidden=192, layers=2, dropout=0.0)
- Training: seq2seq
- Batch size: 32
- Early stopping
- Result (leaderboard): **0.258**
- Notes:
  - Removing dropout did not improve generalisation
  - Similar performance across epochs

---

### GRU (hidden=32, layers=4, dropout=0.1)
- Training: seq2seq
- Batch size: 32
- Early stopping
- Validation (local): weighted Pearson ≈ **0.283**
- Result (leaderboard): **0.2699**
- Notes:
  - Strong improvement despite small hidden size
  - Indicates depth > width for this problem

---

### GRU (hidden=32, layers=6, dropout=0.1)
- Training: seq2seq
- Batch size: 32
- Early stopping
- Result (leaderboard): **0.263**
- Notes:
  - Extra depth without enough capacity hurt performance
  - Likely underfitting

---

### GRU (hidden=64, layers=4, dropout=0.05)
- Training: seq2seq
- Batch size: 32
- Early stopping
- Result (leaderboard): **0.2728**
- Notes:
  - Best result so far
  - Balanced depth and width
  - Confirms seq2seq + moderate depth is effective

---

## Key Takeaways
- Seq2seq training is critical for performance
- Depth (4–6 layers) matters more than very large hidden size
- Moderate hidden size (32–64) generalises better
- Small dropout (0–0.05) works best
- Increasing model size alone does not guarantee improvement

## Current Best
- **GRU(hidden=64, layers=4, dropout=0.05)**
- **Leaderboard score: 0.2728**