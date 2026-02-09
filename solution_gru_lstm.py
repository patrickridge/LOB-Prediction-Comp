# solution.py (submission-safe: no utils import)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# CONFIG
# ======================

GRU_CKPTS = [
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed42_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed123_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed999_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed2024_metricloss1_aug0.pt",
]

LSTM_CKPT = "lstm_best_lstm_stateful_h150_din64_dx1_viewsigmoid_silu_aug1x5_lr0.0003_wd0.1_bs32_seed42.pt"

# Weights: 4 GRUs get 60%, LSTM gets 40%
GRU_WEIGHT = 0.15      # per GRU (4 × 0.15 = 0.6)
LSTM_WEIGHT = 0.40

CLIP_MIN, CLIP_MAX = -6.0, 6.0
DEVICE = "cpu"

# ======================
# Model definitions
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


class HighwayHead(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_in)
        self.W_h = nn.Linear(d_in, d_mid)
        self.W_t = nn.Linear(d_in, d_mid)
        self.W_x = nn.Linear(d_in, d_mid)
        self.W_out = nn.Linear(d_mid, d_out)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x_norm = self.ln(x)
        h = F.gelu(self.W_h(x_norm))
        t = torch.sigmoid(self.W_t(x_norm))
        x_s = self.W_x(x_norm)
        y_mid = t * h + (1 - t) * x_s
        return self.W_out(self.dropout(y_mid))


class StatefulLSTMModel(nn.Module):
    def __init__(self, d_in: int, hidden: int, d_out: int = 2, 
                 feature_view: str = "none", dropout: float = 0.1):
        super().__init__()
        self.hidden = hidden
        self.feature_view = feature_view
        
        if feature_view == "sigmoid_silu":
            lstm_in = 2 * d_in
        else:
            lstm_in = d_in
        
        self.lstm = nn.LSTM(lstm_in, hidden, num_layers=1, batch_first=True)
        self.head = HighwayHead(hidden, hidden, d_out, dropout=dropout)
    
    def forward(self, x, hidden=None):
        if self.feature_view == "sigmoid_silu":
            x = torch.cat([torch.sigmoid(x), F.silu(x)], dim=-1)
        out, hidden = self.lstm(x, hidden)
        pred = self.head(out)
        return pred, hidden


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    5-model ensemble: 4 GRUs (60%) + 1 LSTM (40%)
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Storage for models and hidden states
        self.gru_models = []
        self.gru_hs = []
        
        # Load GRU models
        for name in GRU_CKPTS:
            ckpt_path = os.path.join(base_dir, name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing checkpoint: {name}")

            ckpt = torch.load(ckpt_path, map_location=DEVICE)
            
            model = GRUModel(
                d_in=int(ckpt.get("input_dim", 32)),
                hidden=int(ckpt.get("hidden", 128)),
                d_out=int(ckpt.get("d_out", 2)),
                num_layers=int(ckpt.get("num_layers", 4)),
                dropout=float(ckpt.get("dropout", 0.03)),
            ).to(DEVICE)
            
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            
            self.gru_models.append(model)
            self.gru_hs.append(None)
        
        # Load LSTM model
        lstm_path = os.path.join(base_dir, LSTM_CKPT)
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"Missing checkpoint: {LSTM_CKPT}")
        
        lstm_ckpt = torch.load(lstm_path, map_location=DEVICE)
        
        self.lstm_model = StatefulLSTMModel(
            d_in=int(lstm_ckpt.get("d_in", 64)),
            hidden=int(lstm_ckpt.get("hidden", 150)),
            d_out=2,
            feature_view=str(lstm_ckpt.get("feature_view", "sigmoid_silu")),
            dropout=float(lstm_ckpt.get("dropout", 0.1)),
        ).to(DEVICE)
        
        self.lstm_model.load_state_dict(lstm_ckpt["state_dict"])
        self.lstm_model.eval()
        self.lstm_h = None
        
        # History for LSTM delta computation
        self.prev_state = None
        
        self.current_seq_ix = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        
        # Reset GRU hidden states
        for i in range(len(self.gru_hs)):
            self.gru_hs[i] = None
        
        # Reset LSTM hidden state
        self.lstm_h = None
        
        # Reset previous state
        self.prev_state = None

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        # Get raw state (32 features)
        x_raw = np.asarray(data_point.state, dtype=np.float32)  # [32]
        
        # === GRU predictions (use raw state only) ===
        x_gru = torch.from_numpy(x_raw).to(DEVICE).view(1, 1, -1)  # [1, 1, 32]
        
        gru_preds = []
        for i, model in enumerate(self.gru_models):
            y, self.gru_hs[i] = model(x_gru, self.gru_hs[i])
            gru_preds.append(y[0, 0].cpu().numpy())
        
        # === LSTM predictions (needs deltas) ===
        if self.prev_state is not None:
            dx = x_raw - self.prev_state
        else:
            dx = np.zeros_like(x_raw)
        
        x_with_deltas = np.concatenate([x_raw, dx])  # [64]
        x_lstm = torch.from_numpy(x_with_deltas).to(DEVICE).view(1, 1, -1)  # [1, 1, 64]
        
        lstm_pred, self.lstm_h = self.lstm_model(x_lstm, self.lstm_h)
        lstm_pred = lstm_pred[0, 0].cpu().numpy()
        
        # Update previous state for next delta
        self.prev_state = x_raw.copy()
        
        # === Weighted ensemble ===
        pred = np.zeros(2, dtype=np.float32)
        
        for gru_pred in gru_preds:
            pred += GRU_WEIGHT * gru_pred
        
        pred += LSTM_WEIGHT * lstm_pred
        
        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)