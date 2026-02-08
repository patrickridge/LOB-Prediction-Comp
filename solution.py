# solution.py (submission-safe: no utils import)
import os
import numpy as np
import torch
import torch.nn as nn

# ======================
# CONFIG
# ======================

# Put BOTH checkpoint files next to solution.py in the zip root
CKPT_NAMES = [
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed42_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed999_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed123_metricloss1_aug0.pt",
]

CLIP_MIN, CLIP_MAX = -6.0, 6.0
DEVICE = "cpu"  # submission env is CPU

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
        out, h = self.gru(x, h)
        y = self.head(out)
        return y, h


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    Streaming ensemble GRU inference:
    - maintains hidden state per sequence per model
    - averages predictions across models
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        self.models = []
        self.hs = []  # one hidden state per model

        for name in CKPT_NAMES:
            ckpt_path = os.path.join(base_dir, name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"PRERUN ERROR: Missing checkpoint {name}. "
                    f"Make sure it is included in your submission zip next to solution.py."
                )

            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            d_in = int(ckpt.get("input_dim", DEFAULT_INPUT_DIM))
            d_out = int(ckpt.get("d_out", DEFAULT_D_OUT))
            hidden = int(ckpt.get("hidden", DEFAULT_HIDDEN))
            num_layers = int(ckpt.get("num_layers", DEFAULT_NUM_LAYERS))
            dropout = float(ckpt.get("dropout", DEFAULT_DROPOUT))

            model = GRUModel(
                d_in=d_in,
                hidden=hidden,
                d_out=d_out,
                num_layers=num_layers,
                dropout=dropout,
            ).to(DEVICE)

            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self.models.append(model)
            self.hs.append(None)

        self.current_seq_ix = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        # reset hidden state for each model
        for i in range(len(self.hs)):
            self.hs[i] = None

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)

        preds = []
        for i, model in enumerate(self.models):
            y, self.hs[i] = model(x, self.hs[i])
            preds.append(y[0, 0].cpu().numpy().astype(np.float32))

        pred = np.sum(np.stack(preds, axis=0) * weights[:, None], axis=0)
        pred *= 0.95
        
        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)