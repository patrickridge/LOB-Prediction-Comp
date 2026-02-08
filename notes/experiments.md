## Experiments

### Reference: Baseline GRU (provided ONNX)
Mean Weighted Pearson: 0.2595
- t0: 0.3884
- t1: 0.1306

Notes:
- t1 is consistently weaker than t0.
- Any preprocessing/post-processing must match training, otherwise scores collapse.

## GRU Experiments Summary (Seq2Seq, Streaming Inference)

### Best results
- **HIDDEN=128, LAYERS=4, DROPOUT=0.03** → **0.2732** *(best so far)*
- **HIDDEN=64,  LAYERS=4, DROPOUT=0.05** → **0.2728**
- **HIDDEN=32,  LAYERS=4, DROPOUT=0.10** → **0.2699**

### Mid-tier results
- **HIDDEN=32,  LAYERS=6, DROPOUT=0.10** → **0.2630**
- **HIDDEN=128, LAYERS=2, DROPOUT=0.05** → **~0.2640**
- **HIDDEN=192, LAYERS=2, DROPOUT=0.00** → **0.2581**
- **HIDDEN=256, LAYERS=2, DROPOUT=0.05** → **0.2581**

### Exploratory / weaker configurations
- **HIDDEN=96,  LAYERS=3, DROPOUT=0.02** → **~0.278 (local)**, ~0.279 leaderboard test
- **HIDDEN=96,  LAYERS=4, DROPOUT=0.01** → **0.2743**
- **HIDDEN=128, LAYERS=6, DROPOUT=0.10** → **0.2581**

### Observed patterns
- 4-layer GRUs consistently outperform 2-layer models
- Hidden size sweet spot appears to be **64–128**
- Small dropout (**~0.03–0.06**) improves generalisation
- Too many layers with small hidden (e.g. 32/6) degrades performance
- Larger hidden sizes (192–256) need ≥4 layers to be effective (not yet tested)

### Current best configuration
- **GRU, HIDDEN=128, LAYERS=4, DROPOUT=0.03**
- Weighted Pearson ≈ **0.273**