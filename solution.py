# solution.py (submission-safe: no utils import)

import os
import numpy as np

import torch
import torch.nn as nn

# ======================
# CONFIG (easy to change)
# ======================

# IMPORTANT: the checkpoint file must be in the ZIP ROOT next to solution.py
# i.e. unzip -l submission.zip should show:
#   solution.py
#   gru_best_h32_L4_do0.1.pt
CKPT_NAME = "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed42_metricloss1_aug0.pt"

# Output clipping (competition expectation)
CLIP_MIN, CLIP_MAX = -6.0, 6.0

# Submission environment is CPU-only
DEVICE = "cpu"

# Fallbacks if checkpoint is missing metadata (shouldn't happen, but safe)
DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2
DEFAULT_HIDDEN = 128
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.03


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

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        # x: [B, T, d_in]
        out, h = self.gru(x, h)   # out: [B, T, hidden]
        y = self.head(out)        # y:  [B, T, d_out]
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
                f"PRERUN ERROR: Make sure {CKPT_NAME} is included in your submission zip "
                f"next to solution.py."
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

        self.current_seq_ix = None
        self.h = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        self.h = None

    @torch.no_grad()
    def predict(self, data_point):
        # Competition runtime provides data_point with:
        #   - seq_ix (int)
        #   - state (np array shape [32])
        #   - need_prediction (bool)

        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)      # [d_in]
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)       # [1, 1, d_in]

        y, self.h = self.model(x, self.h)                          # y: [1, 1, d_out]
        pred = y[0, 0].cpu().numpy().astype(np.float32)            # [d_out]

        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)