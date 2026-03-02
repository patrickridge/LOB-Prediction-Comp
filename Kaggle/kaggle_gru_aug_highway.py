# ===== GRU training with AUGMENTATION + HIGHWAY HEAD =====
# Key changes from base GRU cell:
#   1. AUGMENT = True, 5x dataset, scale U(0.6,1.4), noise=0.05σ  ← biggest gain (+0.0035 R²)
#   2. Highway Head instead of linear  ← +0.00116 R²
#
# Paste this as a NEW cell in kaggle_train.ipynb and run.
# Then download the gru_best_*.pt checkpoint.

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Paths
# -------------------------
TRAIN_FILE = Path("/kaggle/input/lob-data/train.parquet")
VALID_FILE  = Path("/kaggle/input/lob-data/valid.parquet")
OUT_DIR     = Path("/kaggle/working")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# RUN CONFIG
# -------------------------
SEED = 42          # change to 999 for a second run

HIDDEN     = 128
NUM_LAYERS = 4
DROPOUT    = 0.03

LR           = 3e-4
WEIGHT_DECAY = 1e-5
BATCH_SIZE   = 32
EPOCHS       = 60
PATIENCE     = 6
CLIP_NORM    = 1.0
NUM_WORKERS  = 2

# ---- Augmentation (THE key improvement) ----
AUGMENT         = True
AUG_MULTIPLIER  = 5           # effective 5x dataset
SCALE_LOW       = 0.6         # 5th place exact params
SCALE_HIGH      = 1.4
NOISE_FRAC      = 0.05        # NOTE: 0.05 not 0.005
AUG_EPS         = 1e-6

INPUT_DIM = 32
D_OUT     = 2

# -------------------------
# Repro
# -------------------------
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

seed_all(SEED)

def _worker_init_fn(worker_id):
    s = SEED + worker_id * 1000
    np.random.seed(s); random.seed(s)

# -------------------------
# Dataset with 5x augmentation
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, df, augment=False, global_feat_std=None):
        self.augment       = augment
        self.aug_mult      = AUG_MULTIPLIER if augment else 1

        self.groups = []
        seq_stds = []
        for _, g in df.groupby("seq_ix", sort=False):
            g = g.sort_values("step_in_seq")
            x = g.iloc[:, 3:35].to_numpy(np.float32)
            y = g.iloc[:, 35:37].to_numpy(np.float32)
            n = g["need_prediction"].to_numpy(np.uint8)
            self.groups.append((x, y, n))
            seq_stds.append(x.std(axis=0, ddof=0))

        if global_feat_std is None:
            self.global_feat_std = np.median(np.stack(seq_stds, 0), axis=0).astype(np.float32)
        else:
            self.global_feat_std = global_feat_std.astype(np.float32)

    def __len__(self):
        return len(self.groups) * self.aug_mult

    def __getitem__(self, idx):
        seq_idx = idx // self.aug_mult
        aug_idx = idx %  self.aug_mult
        x, y, n = self.groups[seq_idx]
        x = x.copy()

        if self.augment and aug_idx > 0:
            # 1. variance normalisation to global median std
            seq_std = x.std(axis=0, ddof=0) + AUG_EPS
            x = x * (self.global_feat_std / seq_std)[None, :]
            # 2. random global scaling U(0.6, 1.4)
            x *= np.float32(np.random.uniform(SCALE_LOW, SCALE_HIGH))
            # 3. gaussian noise at 5% of global std
            x += np.random.normal(0, 1, x.shape).astype(np.float32) * (NOISE_FRAC * self.global_feat_std)[None, :]

        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(n)

def collate_stack(batch):
    xs, ys, ns = zip(*batch)
    return torch.stack(xs, 0), torch.stack(ys, 0), torch.stack(ns, 0)

# -------------------------
# Highway Head (5th place, +0.00116 R²)
# -------------------------
class HighwayHead(nn.Module):
    def __init__(self, d_in, d_mid, d_out, dropout=0.0):
        super().__init__()
        self.ln    = nn.LayerNorm(d_in)
        self.W_h   = nn.Linear(d_in, d_mid)
        self.W_t   = nn.Linear(d_in, d_mid)
        self.W_x   = nn.Linear(d_in, d_mid)
        self.W_out = nn.Linear(d_mid, d_out)
        self.drop  = nn.Dropout(dropout)
        nn.init.constant_(self.W_t.bias, -1.0)  # start biased toward passthrough

    def forward(self, x):
        xn  = self.ln(x)
        h   = F.gelu(self.W_h(xn))
        t   = torch.sigmoid(self.W_t(xn))
        xs  = self.W_x(xn)
        mid = t * h + (1 - t) * xs
        return self.W_out(self.drop(mid))

# -------------------------
# GRU model with Highway Head
# -------------------------
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden, num_layers, dropout, d_out):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers >= 2 else 0.0,
            batch_first=True,
        )
        self.head = HighwayHead(hidden, hidden, d_out, dropout=0.0)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out)

# -------------------------
# Metric loss (weighted Pearson)
# -------------------------
def weighted_pearson_1d(x, y, w, eps=1e-6):
    wsum = w.sum() + eps
    mx   = (w * x).sum() / wsum
    my   = (w * y).sum() / wsum
    xc   = x - mx; yc = y - my
    cov  = (w * xc * yc).sum() / wsum
    vx   = (w * xc * xc).sum() / wsum
    vy   = (w * yc * yc).sum() / wsum
    return cov / (torch.sqrt(vx * vy) + eps)

def metric_loss(pred, target, eps=1e-6):
    w0 = target[:, 0].abs().clamp_min(1e-3)
    w1 = target[:, 1].abs().clamp_min(1e-3)
    c0 = weighted_pearson_1d(pred[:, 0], target[:, 0], w0, eps)
    c1 = weighted_pearson_1d(pred[:, 1], target[:, 1], w1, eps)
    return -0.5 * (c0 + c1)

# -------------------------
# Load data
# -------------------------
print("Loading data...")
train_df = pd.read_parquet(TRAIN_FILE)
valid_df = pd.read_parquet(VALID_FILE)

train_ds = SeqDataset(train_df, augment=True,  global_feat_std=None)
valid_ds = SeqDataset(valid_df, augment=False, global_feat_std=train_ds.global_feat_std)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"),
                          collate_fn=collate_stack,
                          worker_init_fn=_worker_init_fn if NUM_WORKERS > 0 else None)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(DEVICE=="cuda"),
                          collate_fn=collate_stack)

print(f"DEVICE: {DEVICE} | gpu: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'cpu'}")
print(f"Train seqs: {len(train_ds.groups):,} × {AUG_MULTIPLIER} aug = {len(train_ds):,} samples")
print(f"Config: h={HIDDEN} L={NUM_LAYERS} do={DROPOUT} lr={LR} wd={WEIGHT_DECAY} seed={SEED}")
print(f"Aug: scale=[{SCALE_LOW},{SCALE_HIGH}] noise={NOISE_FRAC}σ  (5th place params)")

# -------------------------
# Train
# -------------------------
tag = (f"h{HIDDEN}_L{NUM_LAYERS}_do{DROPOUT}_lr{LR:g}_wd{WEIGHT_DECAY:g}"
       f"_bs{BATCH_SIZE}_seed{SEED}_metricloss1_aug1x{AUG_MULTIPLIER}_highway")
best_path = OUT_DIR / f"gru_best_{tag}.pt"
last_path = OUT_DIR / f"gru_last_{tag}.pt"

model = GRUModel(INPUT_DIM, HIDDEN, NUM_LAYERS, DROPOUT, D_OUT).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_valid_loss   = float("inf")
best_valid_metric = -float("inf")
bad = 0

for epoch in range(EPOCHS):
    model.train()
    tr_sum, tr_n = 0.0, 0
    for X, Y, N in train_loader:
        X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
        pred = model(X)
        mask = N == 1
        if mask.sum() == 0: continue
        loss = metric_loss(pred[mask], Y[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if CLIP_NORM: torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step()
        tr_sum += loss.item(); tr_n += 1

    model.eval()
    vl_sum, vm_sum, vn = 0.0, 0.0, 0
    with torch.no_grad():
        for X, Y, N in valid_loader:
            X, Y, N = X.to(DEVICE), Y.to(DEVICE), N.to(DEVICE)
            pred = model(X)
            mask = N == 1
            if mask.sum() == 0: continue
            vl = metric_loss(pred[mask], Y[mask])
            vl_sum += vl.item(); vm_sum += (-vl).item(); vn += 1

    vl_avg = vl_sum / max(1, vn)
    vm_avg = vm_sum / max(1, vn)

    if vl_avg < best_valid_loss:
        best_valid_loss   = vl_avg
        best_valid_metric = vm_avg
        bad = 0
        torch.save({
            "state_dict": model.state_dict(),
            "input_dim": INPUT_DIM, "hidden": HIDDEN,
            "num_layers": NUM_LAYERS, "dropout": float(DROPOUT),
            "d_out": D_OUT, "epoch": epoch + 1,
            "best_valid_loss": float(best_valid_loss),
            "best_valid_metric": float(best_valid_metric),
            "tag": tag, "seed": int(SEED),
            "augment": True, "aug_multiplier": AUG_MULTIPLIER,
            "scale_low": SCALE_LOW, "scale_high": SCALE_HIGH,
            "noise_frac": NOISE_FRAC, "highway_head": True,
            "loss_name": "metric_loss_weighted_pearson",
        }, best_path)
    else:
        bad += 1

    print(f"ep {epoch+1:02d}: tr={tr_sum/max(1,tr_n):.4f} "
          f"vl={vl_avg:.4f} vm~={vm_avg:.4f} best={best_valid_loss:.4f} bad={bad}/{PATIENCE}")
    if bad >= PATIENCE: break

torch.save({"state_dict": model.state_dict(), "tag": tag}, last_path)
print(f"\nBEST checkpoint: {best_path}")
print(f"Best valid metric (approx corr): {best_valid_metric:.5f}")
