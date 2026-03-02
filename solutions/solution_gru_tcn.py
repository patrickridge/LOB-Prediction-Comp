# solution.py (submission-safe: no utils import)
import os
import re
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# ======================
# CONFIG
# ======================

CKPT_GRU_NAMES = [
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed42_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed123_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed999_metricloss1_aug0.pt",
    "gru_best_h128_L4_do0.03_lr0.0003_wd1e-05_bs32_seed2024_metricloss1_aug0.pt",
]

CKPT_TCN_NAME = "tcn_gru_best_tcn_gru_C48_K5_D4_tdo0.15_H128_L2_gdo0.1_seed2024.pt"

CLIP_MIN, CLIP_MAX = -6.0, 6.0
DEVICE = "cpu"  # submission env is CPU

# weights (not equal; equal would be 0.5 / 0.5)
W_GRU = 0.8
W_TCN = 0.2

DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2

# GRU fallback
DEFAULT_HIDDEN = 128
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT = 0.03

# TCN/TCNGRU fallback
DEFAULT_TCN_WINDOW = 100
DEFAULT_TCN_CHANNELS = 48
DEFAULT_TCN_LEVELS = 4
DEFAULT_TCN_KERNEL = 5
DEFAULT_TCN_DROPOUT = 0.15

DEFAULT_TCNGRU_HIDDEN = 128
DEFAULT_TCNGRU_LAYERS = 2
DEFAULT_TCNGRU_DROPOUT = 0.1


# ======================
# GRU model
# ======================

class GRUModel(nn.Module):
    def __init__(self, d_in, hidden, d_out, num_layers, dropout):
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
        out, h = self.gru(x, h)
        y = self.head(out)
        return y, h


# ======================
# TCN blocks
# ======================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class ConvWrapper(nn.Module):
    """Wraps a single Conv1d — matches checkpoint key pattern net.N.conv.*"""
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size,
                              padding=padding, dilation=dilation)

    def forward(self, x):
        return self.conv(x)


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            ConvWrapper(channels, kernel_size, dilation),  # net.0.conv.*
            Chomp1d(padding),
            nn.ReLU(),
            ConvWrapper(channels, kernel_size, dilation),  # net.3.conv.*
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TCNGRUModel(nn.Module):
    """
    in_proj (1x1 conv) -> TCN blocks -> GRU -> Linear head
    Matches checkpoint key structure exactly.
    """
    def __init__(self, d_in, d_out, tcn_channels, tcn_levels, tcn_kernel, tcn_dropout,
                 gru_hidden, gru_layers, gru_dropout):
        super().__init__()
        self.in_proj = nn.Conv1d(d_in, tcn_channels, 1)
        self.tcn = nn.Sequential(*[
            TemporalBlock(tcn_channels, tcn_kernel, 2 ** i, tcn_dropout)
            for i in range(tcn_levels)
        ])
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(gru_hidden, d_out)

    def forward(self, x):
        x = x.transpose(1, 2)       # [B, T, D] -> [B, D, T]
        x = self.in_proj(x)         # [B, C, T]
        x = self.tcn(x)             # [B, C, T]
        x = x.transpose(1, 2)       # [B, T, C]
        out, _ = self.gru(x)        # [B, T, H]
        return self.head(out[:, -1, :])  # [B, 2]


def _parse_tcn_gru_from_filename(fname):
    """
    Parse patterns like:
    ..._C48_K5_D4_tdo0.15_H128_L2_gdo0.1...
    """
    def find(pattern, cast, default):
        m = re.search(pattern, fname)
        return cast(m.group(1)) if m else default

    C = find(r"_C(\d+)", int, DEFAULT_TCN_CHANNELS)
    K = find(r"_K(\d+)", int, DEFAULT_TCN_KERNEL)
    D = find(r"_D(\d+)", int, DEFAULT_TCN_LEVELS)
    tdo = find(r"_tdo([0-9.]+)", float, DEFAULT_TCN_DROPOUT)
    H = find(r"_H(\d+)", int, DEFAULT_TCNGRU_HIDDEN)
    L = find(r"_L(\d+)", int, DEFAULT_TCNGRU_LAYERS)
    gdo = find(r"_gdo([0-9.]+)", float, DEFAULT_TCNGRU_DROPOUT)
    return C, K, D, tdo, H, L, gdo


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    Ensemble:
      p_gru  = mean(GRU streaming ensemble)
      p_tcn  = TCNGRU(windowed)  (computed only when need_prediction=True)
      pred   = W_GRU * p_gru + W_TCN * p_tcn
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ---- Load GRUs ----
        self.gru_models = []
        self.gru_hs = []

        for name in CKPT_GRU_NAMES:
            ckpt_path = os.path.join(base_dir, name)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError("Missing GRU checkpoint in zip root: %s" % name)

            ckpt = torch.load(ckpt_path, map_location=DEVICE)

            d_in = int(ckpt.get("input_dim", DEFAULT_INPUT_DIM))
            d_out = int(ckpt.get("d_out", DEFAULT_D_OUT))
            hidden = int(ckpt.get("hidden", DEFAULT_HIDDEN))
            num_layers = int(ckpt.get("num_layers", DEFAULT_NUM_LAYERS))
            dropout = float(ckpt.get("dropout", DEFAULT_DROPOUT))

            model = GRUModel(d_in=d_in, hidden=hidden, d_out=d_out, num_layers=num_layers, dropout=dropout).to(DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            self.gru_models.append(model)
            self.gru_hs.append(None)

        # ---- Load TCNGRU ----
        tcn_path = os.path.join(base_dir, CKPT_TCN_NAME)
        if not os.path.exists(tcn_path):
            raise FileNotFoundError("Missing TCN/TCNGRU checkpoint in zip root: %s" % CKPT_TCN_NAME)

        tcn_ckpt = torch.load(tcn_path, map_location=DEVICE)

        # prefer ckpt keys if they exist, else parse filename
        C, K, D, tdo, H, L, gdo = _parse_tcn_gru_from_filename(CKPT_TCN_NAME)

        d_in = int(tcn_ckpt.get("input_dim", DEFAULT_INPUT_DIM))
        d_out = int(tcn_ckpt.get("d_out", DEFAULT_D_OUT))
        self.tcn_window = int(tcn_ckpt.get("window", DEFAULT_TCN_WINDOW))

        tcn_channels = int(tcn_ckpt.get("tcn_channels", tcn_ckpt.get("channels", C)))
        tcn_levels = int(tcn_ckpt.get("tcn_levels", tcn_ckpt.get("levels", D)))
        tcn_kernel = int(tcn_ckpt.get("kernel_size", K))
        tcn_dropout = float(tcn_ckpt.get("tcn_dropout", tcn_ckpt.get("dropout", tdo)))

        gru_hidden = int(tcn_ckpt.get("gru_hidden", tcn_ckpt.get("hidden", H)))
        gru_layers = int(tcn_ckpt.get("gru_layers", tcn_ckpt.get("num_layers", L)))
        gru_dropout = float(tcn_ckpt.get("gru_dropout", gdo))

        self.tcn = TCNGRUModel(
            d_in=d_in, d_out=d_out,
            tcn_channels=tcn_channels, tcn_levels=tcn_levels, tcn_kernel=tcn_kernel, tcn_dropout=tcn_dropout,
            gru_hidden=gru_hidden, gru_layers=gru_layers, gru_dropout=gru_dropout,
        ).to(DEVICE)

        self.tcn.load_state_dict(tcn_ckpt["state_dict"])
        self.tcn.eval()

        # ---- Sequence state ----
        self.current_seq_ix = None
        self.buf = deque(maxlen=self.tcn_window)

    def _reset_seq(self, seq_ix):
        self.current_seq_ix = seq_ix
        for i in range(len(self.gru_hs)):
            self.gru_hs[i] = None
        self.buf.clear()

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)  # [1,1,32]

        # ---- GRU streaming step ----
        preds = []
        for i, m in enumerate(self.gru_models):
            y, self.gru_hs[i] = m(x, self.gru_hs[i])
            preds.append(y[0, 0].cpu().numpy().astype(np.float32))
        p_gru = np.mean(np.stack(preds, axis=0), axis=0)

        # ---- TCN buffer update ----
        self.buf.append(x_np)

        if not bool(data_point.need_prediction):
            return None

        # ---- Window for TCNGRU ----
        if len(self.buf) < self.tcn_window:
            pad = [np.zeros_like(x_np)] * (self.tcn_window - len(self.buf))
            win = pad + list(self.buf)
        else:
            win = list(self.buf)

        Xw = np.asarray(win, dtype=np.float32)               # [T, 32]
        Xw_t = torch.from_numpy(Xw).to(DEVICE).unsqueeze(0)  # [1, T, 32]
        p_tcn = self.tcn(Xw_t)[0].cpu().numpy().astype(np.float32)

        pred = W_GRU * p_gru + W_TCN * p_tcn
        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)