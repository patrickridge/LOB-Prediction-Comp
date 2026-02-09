# solution.py (submission-safe: no utils import)
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# CONFIG
# ======================

LSTM_CKPT = "lstm_best_lstm_stateful_h150_din64_dx1_viewsigmoid_silu_aug1x5_lr0.0003_wd0.1_bs32_seed42.pt"

BILSTM_CKPT = "bilstm_best_bilstmW100_din64_dx1_viewtanh_sigmoid_h256_L2_do0.1_lr0.0003_wd1e-05_bs512_seed42_sub4_vsub2.pt"

# Weights: 50/50
LSTM_WEIGHT = 0.50
BILSTM_WEIGHT = 0.50

CLIP_MIN, CLIP_MAX = -6.0, 6.0
DEVICE = "cpu"

BILSTM_WINDOW = 100

# ======================
# Model definitions
# ======================

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
    2-model ensemble: LSTM (50%) + BiLSTM (50%)
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
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
        
        # Shared history for both models (for deltas and windowing)
        self.window_history = []
        self.prev_state = None
        
        self.current_seq_ix = None

    def _reset_seq(self, seq_ix: int):
        self.current_seq_ix = seq_ix
        
        # Reset LSTM hidden state
        self.lstm_h = None
        
        # Reset window history
        self.window_history = []
        self.prev_state = None

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        # Get raw state (32 features)
        x_raw = np.asarray(data_point.state, dtype=np.float32)  # [32]
        
        # === LSTM predictions (needs deltas) ===
        if self.prev_state is not None:
            dx = x_raw - self.prev_state
        else:
            dx = np.zeros_like(x_raw)
        
        x_with_deltas = np.concatenate([x_raw, dx])  # [64]
        x_lstm = torch.from_numpy(x_with_deltas).to(DEVICE).view(1, 1, -1)  # [1, 1, 64]
        
        lstm_pred, self.lstm_h = self.lstm_model(x_lstm, self.lstm_h)
        lstm_pred = lstm_pred[0, 0].cpu().numpy()
        
        # Update previous state
        self.prev_state = x_raw.copy()
        
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
                    dx_w = np.zeros_like(state)
                else:
                    dx_w = state - window_states[i-1]
                window_with_deltas.append(np.concatenate([state, dx_w]))
            
            x_bilstm = np.stack(window_with_deltas, axis=0)  # [WINDOW, 64]
            x_bilstm = torch.from_numpy(x_bilstm).unsqueeze(0).to(DEVICE)  # [1, WINDOW, 64]
            
            bilstm_pred = self.bilstm_model(x_bilstm)[0].cpu().numpy()
        else:
            # Not enough history - use zero prediction
            bilstm_pred = np.zeros(2, dtype=np.float32)
        
        # === Weighted ensemble ===
        pred = LSTM_WEIGHT * lstm_pred + BILSTM_WEIGHT * bilstm_pred
        
        if not bool(data_point.need_prediction):
            return None

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)