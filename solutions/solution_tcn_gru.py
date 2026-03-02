# solution.py (TCN-GRU only, submission-safe)
import os
import re
from collections import deque
import numpy as np
import torch
import torch.nn as nn

# ======================
# CONFIG
# ======================
CKPT_TCN_NAME = "tcn_gru_best_tcn_gru_C48_K5_D4_tdo0.15_H128_L2_gdo0.1_seed2024.pt"  # <-- change if needed
DEVICE = "cpu"
CLIP_MIN, CLIP_MAX = -6.0, 6.0

DEFAULT_INPUT_DIM = 32
DEFAULT_D_OUT = 2

DEFAULT_TCN_WINDOW = 100
DEFAULT_TCN_CHANNELS = 48
DEFAULT_TCN_LEVELS = 4
DEFAULT_TCN_KERNEL = 5
DEFAULT_TCN_DROPOUT = 0.15

DEFAULT_TCNGRU_HIDDEN = 128
DEFAULT_TCNGRU_LAYERS = 2
DEFAULT_TCNGRU_DROPOUT = 0.1


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


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=1, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNBackbone(nn.Module):
    def __init__(self, d_in, channels, levels, kernel_size, dropout):
        super().__init__()
        layers = []
        in_ch = d_in
        for i in range(levels):
            dil = 2 ** i
            layers.append(TemporalBlock(in_ch, channels, kernel_size, dilation=dil, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B,T,d_in] -> [B,d_in,T]
        x = x.transpose(1, 2)
        return self.tcn(x)  # [B,C,T]


class TCNGRUModel(nn.Module):
    """
    TCN -> GRU -> Linear
    Input:  [B, T, 32]
    Output: [B, 2]
    """
    def __init__(self, d_in, d_out, tcn_channels, tcn_levels, tcn_kernel, tcn_dropout,
                 gru_hidden, gru_layers, gru_dropout):
        super().__init__()
        self.backbone = TCNBackbone(d_in, tcn_channels, tcn_levels, tcn_kernel, tcn_dropout)
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(gru_hidden, d_out)

    def forward(self, x):
        h = self.backbone(x)       # [B,C,T]
        h = h.transpose(1, 2)      # [B,T,C]
        out, _ = self.gru(h)       # [B,T,H]
        y = self.head(out[:, -1])  # [B,2]
        return y


def _parse_from_filename(fname):
    # ..._C48_K5_D4_tdo0.15_H128_L2_gdo0.1...
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
    TCN-GRU only:
    - maintains a rolling window buffer
    - runs model only when need_prediction=True
    """

    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, CKPT_TCN_NAME)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing checkpoint in zip root: {CKPT_TCN_NAME}")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)

        C, K, D, tdo, H, L, gdo = _parse_from_filename(CKPT_TCN_NAME)

        d_in = int(ckpt.get("input_dim", DEFAULT_INPUT_DIM))
        d_out = int(ckpt.get("d_out", DEFAULT_D_OUT))
        self.window = int(ckpt.get("window", DEFAULT_TCN_WINDOW))

        tcn_channels = int(ckpt.get("tcn_channels", ckpt.get("channels", C)))
        tcn_levels = int(ckpt.get("tcn_levels", ckpt.get("levels", D)))
        tcn_kernel = int(ckpt.get("kernel_size", K))
        tcn_dropout = float(ckpt.get("tcn_dropout", ckpt.get("dropout", tdo)))

        gru_hidden = int(ckpt.get("gru_hidden", ckpt.get("hidden", H)))
        gru_layers = int(ckpt.get("gru_layers", ckpt.get("num_layers", L)))
        gru_dropout = float(ckpt.get("gru_dropout", gdo))

        self.model = TCNGRUModel(
            d_in=d_in, d_out=d_out,
            tcn_channels=tcn_channels, tcn_levels=tcn_levels, tcn_kernel=tcn_kernel, tcn_dropout=tcn_dropout,
            gru_hidden=gru_hidden, gru_layers=gru_layers, gru_dropout=gru_dropout,
        ).to(DEVICE)

        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

        self.current_seq_ix = None
        self.buf = deque(maxlen=self.window)

    def _reset_seq(self, seq_ix):
        self.current_seq_ix = seq_ix
        self.buf.clear()

    @torch.no_grad()
    def predict(self, data_point):
        seq_ix = int(data_point.seq_ix)
        if self.current_seq_ix != seq_ix:
            self._reset_seq(seq_ix)

        x_np = np.asarray(data_point.state, dtype=np.float32)
        self.buf.append(x_np)

        if not bool(data_point.need_prediction):
            return None

        # pad left if needed
        if len(self.buf) < self.window:
            pad = [np.zeros_like(x_np)] * (self.window - len(self.buf))
            win = pad + list(self.buf)
        else:
            win = list(self.buf)

        Xw = np.asarray(win, dtype=np.float32)              # [T,32]
        Xw_t = torch.from_numpy(Xw).unsqueeze(0).to(DEVICE) # [1,T,32]

        pred = self.model(Xw_t)[0].cpu().numpy().astype(np.float32)
        return np.clip(pred, CLIP_MIN, CLIP_MAX).astype(np.float32)