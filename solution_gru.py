import os
import sys
import numpy as np
import torch
import torch.nn as nn

# utils live in competition_package
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, "competition_package"))
from utils import DataPoint, ScorerStepByStep

# NOTE: submission env is CPU-only; local can use mps
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


class GRUModel(nn.Module):
    def __init__(self, d_in=32, hidden=256, d_out=2, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x, h=None):
        # x: [B, T, 32]
        out, h = self.gru(x, h)      # out: [B, T, H]
        y = self.head(out)           # y:  [B, T, 2]
        return y, h


class PredictionModel:
    """
    Streaming GRU inference:
    - maintains hidden state per sequence
    - updates on every step
    - returns pred only when need_prediction=True
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, "artifacts", "gru_h256_L2_do0.1.pt")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        hidden = int(ckpt.get("hidden", 256))
        num_layers = int(ckpt.get("num_layers", 2))
        dropout = float(ckpt.get("dropout", 0.1))

        self.model = GRUModel(hidden=hidden, num_layers=num_layers, dropout=dropout).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.current_seq_ix = None
        self.h = None  # hidden state

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.h = None

    @torch.no_grad()
    def predict(self, data_point: DataPoint):
        if self.current_seq_ix != data_point.seq_ix:
            self._reset_seq(data_point.seq_ix)

        x_np = data_point.state.astype(np.float32, copy=False)
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)  # [1,1,32]

        y, self.h = self.model(x, self.h)          # y: [1,1,2]
        pred = y[0, 0].detach().cpu().numpy()      # [2,]

        if not data_point.need_prediction:
            return None

        pred = np.clip(pred, -6.0, 6.0).astype(np.float32)
        return pred


if __name__ == "__main__":
    test_file = os.path.join(CURRENT_DIR, "competition_package", "datasets", "valid.parquet")
    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print("Scoring GRU solution on valid.parquet (Weighted Pearson)...")
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
    for t in scorer.targets:
        print(f"  {t}: {results[t]:.6f}")