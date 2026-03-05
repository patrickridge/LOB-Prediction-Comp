# LOB Predictorium — Wunder Challenge 2

Deep learning solution to short-horizon price movement prediction on anonymised limit order book (LOB) data. Uses GRU-based recurrent neural networks trained with a metric-aligned loss function, ensembled across multiple seeds and architectures (LSTM, BiLSTM). Achieved a leaderboard score of **0.2883** during the competition, and a final out-of-sample score of **0.3116** — placing **42nd globally** in the finals (top 1%) out of 4,919 participants.

## Competition

[Wunder Predictorium](https://wundernn.io/predictorium) — **Task:** Predict two targets per timestep:
- **t0** — short-horizon price movement (stronger signal)
- **t1** — longer-horizon price movement (noisier, main bottleneck)

**Data:** 10,721 training sequences, 1,444 validation sequences. Each sequence is 1,000 timesteps (steps 0–98 warm-up; steps 99–999 scored). 32 engineered LOB + trade features.

**Metric:** Weighted Pearson Correlation — rewards correct direction and relative magnitude; penalises overconfident scale errors.

**Reference baseline (provided ONNX):** 0.2595

## Results

| Stage | Score | Placement |
|-------|-------|-----------|
| Competition leaderboard | 0.2883 | 110 / 4,919 |
| **Finals (out-of-sample)** | **0.3116** | **42nd / 4,919 (top 1%)** |

Best submission: 4-seed GRU ensemble (seeds 42, 123, 999, 2024)

| Setting | Value |
|---------|-------|
| Architecture | GRU, Seq2Seq, streaming inference |
| Hidden size | 128 |
| Layers | 4 |
| Dropout | 0.03 |
| Loss | Negative Pearson correlation |
| LR | 3e-4 |
| Weight decay | 1e-5 |

## Key Insight

Training with MSE produces good validation loss but poor leaderboard alignment. **Training directly on negative Pearson correlation** closes the gap between local validation and leaderboard scores. This single change was the largest performance driver.

## Solution Files

All submission-ready solutions are in the [`solutions/`](solutions/) directory.

| File | Architecture | Notes |
|------|-------------|-------|
| `solution_gru.py` | 4× GRU | Best submission — 0.2883 LB / 0.3116 finals |
| `solution_gru_lstm.py` | GRU + LSTM | Architecture diversity ensemble |
| `solution_gru_bilstm.py` | GRU + BiLSTM | Architecture diversity ensemble |
| `solution_bilstm_lstm.py` | BiLSTM + LSTM | No GRU variant |
| `solution_6_gru_lstm_bilstm.py` | GRU + LSTM + BiLSTM (6-model) | Expanded ensemble |
| `solution_adaptive.py` | GRU (adaptive weighting) | Online adaptive inference |

## Repo Structure

```
├── solutions/             # Submission-ready solution files
├── notes/                 # Experiment logs and observations
│   ├── experiments.md     # Chronological experiment log with LB scores
│   ├── notes.md           # Core model summary and key findings
│   ├── improvements.md    # Ranked improvement ideas
│   └── training.md        # Training diagnostics guide
├── kaggle/                # Kaggle training environment setup
│   ├── KAGGLE.md          # Setup instructions
│   └── kaggle_train.ipynb # Training notebook (GPU)
├── artifacts/             # Feature metadata (feature_names.json, feature_meta.json)
├── data_inspection/       # EDA scripts and visualisations
├── models/                # Trained model weights (gitignored)
└── eval_local.py          # Local evaluation script
```

## Training

See [`kaggle/KAGGLE.md`](kaggle/KAGGLE.md) for environment setup, dataset upload, and checkpoint management on Kaggle (T4×2 GPU).
