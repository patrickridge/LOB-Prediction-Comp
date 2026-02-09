# solution.py (submission-safe: no utils import)
import os
import numpy as np
import torch
import torch.nn as nn

# ======================
# CONFIG
# ======================

GRU_CKPTS = [
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed42_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed123_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed999_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed2024_metricloss1_aug0.pt",
]

BILSTM_CKPT = "bilstm_best_bilstmW100_din64_dx1_viewtanh_sigmoid_h256_L2_do0.1_lr0.0003_wd1e-05_bs512_seed42_sub4_vsub2.pt"

# Weights: 4 GRUs get 50%, BiLSTM gets 50%
GRU_WEIGHT = 0.125      # per GRU (4 × 0.125 = 0.5)
BILSTM_WEIGHT = 0.50

CLIP_MIN, CLIP_MAX = -6.0, 6.0
DEVICE = "cpu"

BILSTM_WINDOW = 100

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


class BiLSTMWindowModel(nn.Module):
    def __init__(self, d_in: int, hidden: int, num_layers: int, dropout: float, 
                 d_out: int = 2, feature_view: str = "none"):
        super().__init__()
        self.feature_view = feature_view
        
        if feature_view == "tanh_sigmoid":
            lstm_in = 2 * d_in
        else:
            lstm_in = d_in
        
        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers >= 2 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * hidden, d_out)
    
    def forward(self, x):
        if self.feature_view == "tanh_sigmoid":
            x = torch.cat([torch.tanh(x), torch.sigmoid(x)], dim=-1)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    5-model ensemble: 4 GRUs (50%) + 1 BiLSTM (50%)
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
        
        # Load BiLSTM model
        bilstm_path = os.path.join(base_dir, BILSTM_CKPT)
        if not os.path.exists(bilstm_path):
            raise FileNotFoundError(f"Missing checkpoint: {BILSTM_CKPT}")
        
        bilstm_ckpt = torch.load(bilstm_path, map_location=DEVICE)
        
        self.bilstm_model = BiLSTMWindowModel(
            d_in=int(bilstm_ckpt.get("d_in", 64)),
            hidden=int(bilstm_ckpt.get("hidden", 256)),
            num_layers=int(bilstm_ckpt.get("num_layers", 2)),
            dropout=float(bilstm_ckpt.get("dropout", 0.1)),
            d_out=2,
            feature_view=str(bilstm_ckpt.get("feature_view", "tanh_sigmoid")),
        ).to(DEVICE)
        
        self.bilstm_model.load_state_dict(bilstm_ckpt["state_dict"])
        self.bilstm_model.eval()
        
        # BiLSTM window history
        self.window_history = []
        
        self.current_seq_ix = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        
        # Reset GRU hidden states
        for i in range(len(self.gru_hs)):
            self.gru_hs[i] = None
        
        # Reset BiLSTM window history
        self.window_history = []

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        # Get raw state (32 features)
        x_raw = np.asarray(data_point.state, dtype=np.float32)  # [32]
        
        # === GRU predictions ===
        x_gru = torch.from_numpy(x_raw).to(DEVICE).view(1, 1, -1)  # [1, 1, 32]
        
        gru_preds = []
        for i, model in enumerate(self.gru_models):
            y, self.gru_hs[i] = model(x_gru, self.gru_hs[i])
            gru_preds.append(y[0, 0].cpu().numpy())
        
        # === BiLSTM predictions ===
        # Add current state to window history
        self.window_history.append(x_raw.copy())
        
        if len(self.window_history) >= BILSTM_WINDOW:
            # Get last WINDOW states
            window_states = self.window_history[-BILSTM_WINDOW:]
            
            # Compute deltas for BiLSTM window
            window_with_deltas = []
            for i, state in enumerate(window_states):
                if i == 0:
                    dx = np.zeros_like(state)
                else:
                    dx = state - window_states[i-1]
                window_with_deltas.append(np.concatenate([state, dx]))
            
            x_bilstm = np.stack(window_with_deltas, axis=0)  # [WINDOW, 64]
            x_bilstm = torch.from_numpy(x_bilstm).unsqueeze(0).to(DEVICE)  # [1, WINDOW, 64]
            
            bilstm_pred = self.bilstm_model(x_bilstm)[0].cpu().numpy()
        else:
            # Not enough history - use zero prediction
            bilstm_pred = np.zeros(2, dtype=np.float32)
        
        # === Weighted ensemble ===
        pred = np.zeros(2, dtype=np.float32)
        
        for gru_pred in gru_preds:
            pred += GRU_WEIGHT * gru_pred
        
        pred += BILSTM_WEIGHT * bilstm_pred
        
        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)