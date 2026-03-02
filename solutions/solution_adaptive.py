# solution_adaptive.py
# 4-GRU ensemble + TCN-GRU + online adaptive weighting
#
# Same checkpoints as solution.py.
# Adds a lightweight online meta-learner (from 5th place report):
#   W_i *= exp(-ETA * (pred_i - ensemble_mean)^2)  [proxy for per-model error]
# Weights are renormalized each step and reset per sequence.
#
# No additional training required — drop in replacement for solution.py.

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
DEVICE = "cpu"

# Static fallback weights (match solution.py: GRU=80%, TCN=20%)
W_GRU_STATIC = 0.8   # shared across all 4 GRUs (0.2 each)
W_TCN_STATIC = 0.2

# Online meta-learner adaptation rate (from 5th place report)
ETA = 0.1

# GRU fallback
DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2
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
# Models
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
        return self.head(out), h


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class ConvWrapper(nn.Module):
    """Wraps Conv1d to match checkpoint key pattern net.N.conv.*"""
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
            ConvWrapper(channels, kernel_size, dilation),
            Chomp1d(padding),
            nn.ReLU(),
            ConvWrapper(channels, kernel_size, dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TCNGRUModel(nn.Module):
    """in_proj (1x1 conv) -> TCN -> GRU -> Linear. Matches checkpoint exactly."""
    def __init__(self, d_in, d_out, tcn_channels, tcn_levels, tcn_kernel, tcn_dropout,
                 gru_hidden, gru_layers, gru_dropout):
        super().__init__()
        self.in_proj = nn.Conv1d(d_in, tcn_channels, 1)
        self.tcn = nn.Sequential(*[
            TemporalBlock(tcn_channels, tcn_kernel, 2 ** i, tcn_dropout)
            for i in range(tcn_levels)
        ])
        self.gru = nn.GRU(tcn_channels, gru_hidden, gru_layers,
                          dropout=gru_dropout if gru_layers >= 2 else 0.0, batch_first=True)
        self.head = nn.Linear(gru_hidden, d_out)

    def forward(self, x):
        x = self.in_proj(x.transpose(1, 2))  # [B,T,D] -> [B,C,T]
        x = self.tcn(x).transpose(1, 2)      # [B,T,C]
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def _parse_tcn_gru(fname):
    def find(pat, cast, default):
        m = re.search(pat, fname)
        return cast(m.group(1)) if m else default
    return (find(r"_C(\d+)", int, DEFAULT_TCN_CHANNELS),
            find(r"_K(\d+)", int, DEFAULT_TCN_KERNEL),
            find(r"_D(\d+)", int, DEFAULT_TCN_LEVELS),
            find(r"_tdo([0-9.]+)", float, DEFAULT_TCN_DROPOUT),
            find(r"_H(\d+)", int, DEFAULT_TCNGRU_HIDDEN),
            find(r"_L(\d+)", int, DEFAULT_TCNGRU_LAYERS),
            find(r"_gdo([0-9.]+)", float, DEFAULT_TCNGRU_DROPOUT))


# ======================
# Submission entrypoint
# ======================

class PredictionModel:
    """
    4-GRU streaming ensemble + TCN-GRU windowed model.
    Adds online adaptive weighting: per-model weights are adjusted
    each step based on deviation from the ensemble mean (proxy for error).
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # ---- Load GRUs ----
        self.gru_models = []
        self.gru_hs = []
        for name in CKPT_GRU_NAMES:
            ckpt = torch.load(os.path.join(base_dir, name), map_location=DEVICE)
            m = GRUModel(
                d_in=int(ckpt.get("input_dim", DEFAULT_INPUT_DIM)),
                hidden=int(ckpt.get("hidden", DEFAULT_HIDDEN)),
                d_out=int(ckpt.get("d_out", DEFAULT_D_OUT)),
                num_layers=int(ckpt.get("num_layers", DEFAULT_NUM_LAYERS)),
                dropout=float(ckpt.get("dropout", DEFAULT_DROPOUT)),
            ).to(DEVICE)
            m.load_state_dict(ckpt["state_dict"])
            m.eval()
            self.gru_models.append(m)
            self.gru_hs.append(None)

        n_gru = len(self.gru_models)

        # ---- Load TCN-GRU ----
        tcn_ckpt = torch.load(os.path.join(base_dir, CKPT_TCN_NAME), map_location=DEVICE)
        C, K, D, tdo, H, L, gdo = _parse_tcn_gru(CKPT_TCN_NAME)
        self.tcn_window = int(tcn_ckpt.get("window", DEFAULT_TCN_WINDOW))
        self.tcn = TCNGRUModel(
            d_in=int(tcn_ckpt.get("input_dim", DEFAULT_INPUT_DIM)),
            d_out=int(tcn_ckpt.get("d_out", DEFAULT_D_OUT)),
            tcn_channels=int(tcn_ckpt.get("tcn_channels", tcn_ckpt.get("channels", C))),
            tcn_levels=int(tcn_ckpt.get("tcn_levels", tcn_ckpt.get("levels", D))),
            tcn_kernel=int(tcn_ckpt.get("kernel_size", K)),
            tcn_dropout=float(tcn_ckpt.get("tcn_dropout", tcn_ckpt.get("dropout", tdo))),
            gru_hidden=int(tcn_ckpt.get("gru_hidden", tcn_ckpt.get("hidden", H))),
            gru_layers=int(tcn_ckpt.get("gru_layers", tcn_ckpt.get("num_layers", L))),
            gru_dropout=float(tcn_ckpt.get("gru_dropout", gdo)),
        ).to(DEVICE)
        self.tcn.load_state_dict(tcn_ckpt["state_dict"])
        self.tcn.eval()

        # ---- Adaptive weight state ----
        # Indices: 0..n_gru-1 = GRU models, n_gru = TCN
        n_models = n_gru + 1
        # Initialize with static weights matching solution.py
        init_w = np.array(
            [W_GRU_STATIC / n_gru] * n_gru + [W_TCN_STATIC],
            dtype=np.float32
        )
        self._init_weights = init_w.copy()
        self._weights = init_w.copy()  # live weights, reset per sequence

        self.current_seq_ix = None
        self.buf = deque(maxlen=self.tcn_window)

    def _reset_seq(self, seq_ix):
        self.current_seq_ix = seq_ix
        for i in range(len(self.gru_hs)):
            self.gru_hs[i] = None
        self.buf.clear()
        self._weights = self._init_weights.copy()

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)
        x = torch.from_numpy(x_np).to(DEVICE).view(1, 1, -1)

        # ---- GRU streaming predictions ----
        gru_preds = []
        for i, m in enumerate(self.gru_models):
            y, self.gru_hs[i] = m(x, self.gru_hs[i])
            gru_preds.append(y[0, 0].cpu().numpy().astype(np.float32))

        self.buf.append(x_np)

        if not bool(data_point.need_prediction):
            return None

        # ---- TCN-GRU windowed prediction ----
        if len(self.buf) < self.tcn_window:
            pad = [np.zeros_like(x_np)] * (self.tcn_window - len(self.buf))
            win = pad + list(self.buf)
        else:
            win = list(self.buf)
        Xw = torch.from_numpy(np.asarray(win, dtype=np.float32)).to(DEVICE).unsqueeze(0)
        tcn_pred = self.tcn(Xw)[0].cpu().numpy().astype(np.float32)

        all_preds = np.stack(gru_preds + [tcn_pred], axis=0)  # [n_models, 2]
        w = self._weights[:, None]  # [n_models, 1]

        # ---- Weighted ensemble ----
        pred = (w * all_preds).sum(axis=0)  # [2]

        # ---- Update weights (online adaptive meta-learner) ----
        # Proxy: penalise models that deviate far from the current weighted ensemble
        ensemble_mean = pred  # use current prediction as reference
        sq_err = np.sum((all_preds - ensemble_mean[None, :]) ** 2, axis=1)  # [n_models]
        self._weights = self._weights * np.exp(-ETA * sq_err)
        self._weights = self._weights / (self._weights.sum() + 1e-12)

        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)
