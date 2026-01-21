# solution.py (submission-safe: no utils import)

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ======================
# CONFIG (easy to change)
# ======================

# Put your checkpoint in the zip alongside solution.py
CKPT_NAME = "gru_best_h192_L2_do0.pt"

# If ckpt is missing fields, fall back to these defaults
DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2
DEFAULT_HIDDEN = 192
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0

# Output clipping (competition expectation)
CLIP_MIN, CLIP_MAX = -6.0, 6.0

# Submission env is CPU-only. Keep CPU here to be safe.
DEVICE = "cpu"


# ======================
# Model definition
# ======================

class GRUModel(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int, num_layers: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, d_in]
        out, h = self.gru(x, h)     # out: [B, T, hidden]
        y = self.head(out)          # y: [B, T, d_out]
        return y, h


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    Streaming GRU inference:
    - maintains hidden state per sequence (seq_ix)
    - updates each step
    - returns prediction only when need_prediction=True, else None
    """
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, CKPT_NAME)

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                f"Make sure {CKPT_NAME} is included in your submission zip next to solution.py."
            )

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        d_in = int(ckpt.get("input_dim", DEFAULT_INPUT_DIM))
        d_out = int(ckpt.get("d_out", DEFAULT_D_OUT))
        hidden = int(ckpt.get("hidden", DEFAULT_HIDDEN))
        num_layers = int(ckpt.get("num_layers", DEFAULT_NUM_LAYERS))
        dropout = float(ckpt.get("dropout", DEFAULT_DROPOUT))

        self.model = GRUModel(
            d_in=d_in,
            hidden=hidden,
            d_out=d_out,
            num_layers=num_layers,
            dropout=dropout,
        ).to(DEVICE)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.current_seq_ix: Optional[int] = None
        self.h: Optional[torch.Tensor] = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.h = None

    @torch.no_grad()
    def predict(self, data_point: Any):
        """
        data_point is provided by the competition runtime.
        It is expected to have:
          - data_point.seq_ix (int)
          - data_point.state (np array shape [32])
          - data_point.need_prediction (bool)
        """
        seq_ix = int(data_point.seq_ix)

        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)  # [32]
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)   # [1,1,32]

        y, self.h = self.model(x, self.h)                      # y: [1,1,2]
        pred = y[0, 0].cpu().numpy().astype(np.float32)        # [2]

        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)


# ======================
# Optional local test (won't run in submission)
# ======================

if __name__ == "__main__":
    # Only for your local sanity check if you have utils.py available locally.
    # This block is NOT executed by the submission server.
    try:
        from utils import ScorerStepByStep  # type: ignore
    except Exception as e:
        print("Local test skipped (utils.py not found). Error:", e)
        raise SystemExit(0)

    # Adjust this path to your local valid parquet if needed
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "competition_package", "datasets", "valid.parquet")

    model = PredictionModel()
    scorer = ScorerStepByStep(test_file)

    print(f"DEVICE={DEVICE} | CKPT={CKPT_NAME}")
    print("Scoring on valid.parquet...")
    results = scorer.score(model)
    print("Results:", results)