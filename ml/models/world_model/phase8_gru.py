"""
PHASE 8 – GRU World Model
==========================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

DESIGN
------
A multi-task GRU that jointly learns three objectives from temporal sequences:

  Task 1 (World Model core): Next-state regression
      Input  : [S_{t-L+1}, ..., S_t]   shape (batch, L, D)
      Output : Ŝ_{t+1}                  shape (batch, D)
      Loss   : MSELoss  (predicts how the network state evolves)

  Task 2: Binary attack classification
      Output : P(y=ATTACK | sequence)   shape (batch, 2)
      Loss   : CrossEntropyLoss (class-weighted)

  Task 3: Multiclass attack classification
      Output : P(class=k | sequence)    shape (batch, 15)
      Loss   : CrossEntropyLoss (class-weighted)

  Total loss = w1*L_regression + w2*L_binary + w3*L_multiclass
               0.20              0.40           0.40

ARCHITECTURE
------------
  GRU(input=60, hidden=128, layers=2, dropout=0.3, batch_first=True)
  → Last timestep hidden state (batch, 128)
  → LayerNorm(128)
  → Head 1: Linear(128→60)           [state regression]
  → Head 2: Linear(128→64)→ReLU→Linear(64→2)   [binary]
  → Head 3: Linear(128→64)→ReLU→Linear(64→15)  [multiclass]

TRAINING
--------
  Device   : MPS (Apple Silicon) → fallback CPU
  Optimizer: Adam(lr=1e-3, weight_decay=1e-5)
  Scheduler: ReduceLROnPlateau(patience=2, factor=0.5)
  Epochs   : 20, early stopping patience=5 (on val binary F1)
  Batch    : 1024 (train), 2048 (eval)
  Scaler   : StandardScaler from Phase 6 (applied inside Dataset)

STRICT RULES:
  - Scaler fit ONLY on train (loaded from Phase 6)
  - Train/Val/Test strictly separated
  - All metrics from real model outputs — ZERO fabrication
  - Fixed seed = 42
"""

import os, sys, json, warnings, time, gc, pickle, copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix, average_precision_score
)
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
SEQ_DIR     = PROC_DIR / "sequences"
MODELS_DIR  = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"
WM_DIR      = BASE_DIR / "models" / "world_model"
WM_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN      = 10
HIDDEN_DIM   = 128
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_TRAIN  = 1024
BATCH_EVAL   = 2048
LR           = 1e-3
WD           = 1e-5
EPOCHS       = 20
PATIENCE     = 5
W_REGR       = 0.20    # regression loss weight
W_BIN        = 0.40    # binary classification loss weight
W_MC         = 0.40    # multiclass classification loss weight
N_CLASSES_MC = 15


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class NetworkSequenceDataset(Dataset):
    """
    Index-based temporal sequence dataset (memory-efficient).
    Builds sequences [S_{t-L+1}...S_t] on-the-fly from sorted DataFrame.
    Applies scaler inside __getitem__ so no extra memory for scaled copy.
    Boundary rule: sequences never cross day_order boundaries.
    """
    def __init__(self, df: pd.DataFrame, seq_len: int, feature_cols: list,
                 scaler=None, stride: int = 1):
        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.next_cols    = [f"next_{c}" for c in feature_cols]
        self.D            = len(feature_cols)

        # Extract raw numpy arrays
        feat_arr      = df[feature_cols].values.astype(np.float32)
        next_feat_arr = df[self.next_cols].values.astype(np.float32)

        # Apply scaler if provided
        if scaler is not None:
            feat_arr      = scaler.transform(feat_arr).astype(np.float32)
            next_feat_arr = scaler.transform(next_feat_arr).astype(np.float32)

        self.feat_arr      = feat_arr
        self.next_feat_arr = next_feat_arr
        self.y_bin_arr     = df["next_label_binary"].values.astype(np.int64)
        self.y_mc_arr      = df["next_label_multiclass"].values.astype(np.int64)

        # Build valid start indices (no cross-day sequences)
        day_orders  = df["day_order"].values
        is_boundary = df["is_boundary"].values
        n = len(df)
        valid_starts = []
        i = 0
        while i + seq_len < n:
            end = i + seq_len
            if day_orders[i] != day_orders[end]:
                i += 1; continue
            if np.any(is_boundary[i:end]):
                i += 1; continue
            valid_starts.append(i)
            i += stride
        self.indices = np.array(valid_starts, dtype=np.int64)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.seq_len
        X       = torch.from_numpy(self.feat_arr[start:end].copy())
        y_state = torch.from_numpy(self.next_feat_arr[end - 1].copy())
        y_bin   = torch.tensor(int(self.y_bin_arr[end - 1]), dtype=torch.long)
        y_mc    = torch.tensor(int(self.y_mc_arr[end - 1]),  dtype=torch.long)
        return X, y_state, y_bin, y_mc


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
class GRUWorldModel(nn.Module):
    """
    Multi-task GRU World Model.

    Architecture:
      GRU(D→hidden, layers, dropout) → last hidden state
      → LayerNorm → 3 task heads
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 n_classes_mc: int, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Head 1: next-state regression
        self.head_state = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        # Head 2: binary classification
        self.head_binary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # Head 3: multiclass classification
        self.head_mc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes_mc)
        )

    def forward(self, x):
        # x: (batch, seq_len, D)
        out, _ = self.gru(x)          # (batch, seq_len, hidden)
        last   = out[:, -1, :]         # (batch, hidden)
        last   = self.layer_norm(last)
        return self.head_state(last), self.head_binary(last), self.head_mc(last)

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(y_array, n_classes, device):
    """Inverse-frequency class weights for CrossEntropy."""
    counts = np.bincount(y_array, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)   # avoid div/0
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes   # normalize
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, criterion_state, criterion_bin, criterion_mc,
             device, split_name="Val"):
    """Evaluate model — returns loss components + classification metrics."""
    model.eval()
    all_y_bin, all_p_bin, all_prob_bin = [], [], []
    all_y_mc,  all_p_mc               = [], []
    total_loss = total_l_s = total_l_b = total_l_m = 0.0
    n_batches = 0

    with torch.no_grad():
        for X, y_state, y_bin, y_mc in loader:
            X       = X.to(device)
            y_state = y_state.to(device)
            y_bin   = y_bin.to(device)
            y_mc    = y_mc.to(device)

            pred_state, pred_bin, pred_mc = model(X)

            l_s = criterion_state(pred_state, y_state)
            l_b = criterion_bin(pred_bin, y_bin)
            l_m = criterion_mc(pred_mc, y_mc)
            loss = W_REGR*l_s + W_BIN*l_b + W_MC*l_m

            total_loss += loss.item();  total_l_s += l_s.item()
            total_l_b  += l_b.item();  total_l_m += l_m.item()
            n_batches  += 1

            all_y_bin.extend(y_bin.cpu().numpy())
            all_p_bin.extend(pred_bin.argmax(1).cpu().numpy())
            all_prob_bin.extend(F.softmax(pred_bin, dim=1)[:, 1].cpu().numpy())
            all_y_mc.extend(y_mc.cpu().numpy())
            all_p_mc.extend(pred_mc.argmax(1).cpu().numpy())

    all_y_bin = np.array(all_y_bin);  all_p_bin = np.array(all_p_bin)
    all_prob_bin = np.array(all_prob_bin)
    all_y_mc  = np.array(all_y_mc);   all_p_mc  = np.array(all_p_mc)

    metrics = {
        "loss":           total_loss / n_batches,
        "loss_regression":total_l_s  / n_batches,
        "loss_binary":    total_l_b  / n_batches,
        "loss_multiclass":total_l_m  / n_batches,
        # Binary
        "bin_accuracy":   float(accuracy_score(all_y_bin, all_p_bin)),
        "bin_f1_macro":   float(f1_score(all_y_bin, all_p_bin, average="macro",    zero_division=0)),
        "bin_f1_weighted":float(f1_score(all_y_bin, all_p_bin, average="weighted", zero_division=0)),
        "bin_recall_macro":float(recall_score(all_y_bin, all_p_bin, average="macro", zero_division=0)),
        "bin_precision_macro":float(precision_score(all_y_bin, all_p_bin, average="macro", zero_division=0)),
        # Multiclass
        "mc_accuracy":    float(accuracy_score(all_y_mc, all_p_mc)),
        "mc_f1_macro":    float(f1_score(all_y_mc, all_p_mc, average="macro",    zero_division=0)),
        "mc_f1_weighted": float(f1_score(all_y_mc, all_p_mc, average="weighted", zero_division=0)),
        "mc_recall_macro":float(recall_score(all_y_mc, all_p_mc, average="macro", zero_division=0)),
    }
    # ROC-AUC binary (only if both classes present)
    if len(np.unique(all_y_bin)) == 2:
        metrics["bin_roc_auc"]       = float(roc_auc_score(all_y_bin, all_prob_bin))
        metrics["bin_avg_precision"] = float(average_precision_score(all_y_bin, all_prob_bin))

    return metrics, all_y_bin, all_p_bin, all_y_mc, all_p_mc


def plot_training_curves(history, path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    metrics_to_plot = [
        ("loss",           "Total Loss",       "#e74c3c"),
        ("loss_binary",    "Binary CE Loss",   "#e67e22"),
        ("loss_multiclass","Multiclass CE",    "#8e44ad"),
        ("bin_f1_macro",   "Binary F1-macro",  "#27ae60"),
        ("bin_roc_auc",    "Binary ROC-AUC",   "#2980b9"),
        ("mc_f1_macro",    "Multiclass F1-mac","#16a085"),
    ]
    for ax, (key, title, color) in zip(axes.flatten(), metrics_to_plot):
        train_vals = [h["train"].get(key, None) for h in history]
        val_vals   = [h["val"].get(key, None)   for h in history]
        epochs_x   = list(range(1, len(history)+1))
        if any(v is not None for v in train_vals):
            ax.plot(epochs_x, [v if v else 0 for v in train_vals],
                    label="Train", color=color, lw=2, alpha=0.7)
        if any(v is not None for v in val_vals):
            ax.plot(epochs_x, [v if v else 0 for v in val_vals],
                    label="Val", color=color, lw=2, linestyle="--")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.suptitle("GRU World Model — Training History", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 8 – GRU WORLD MODEL (MULTI-TASK)")
    print("="*72)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")

    # ── Load definitions ──────────────────────────────────────────────────
    print("\n[STEP 1] Loading definitions...")
    with open(PROC_DIR/"feature_list.json")   as f: feat_def  = json.load(f)
    with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)
    with open(MODELS_DIR/"standard_scaler.pkl","rb") as f:
        scaler = pickle.load(f)["scaler"]

    FEATURE_COLS = feat_def["all_numeric_features"]
    CLASS_NAMES  = label_enc["classes"]
    D = len(FEATURE_COLS)
    print(f"  D={D}, seq_len={SEQ_LEN}, classes={N_CLASSES_MC}")
    print(f"  hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}")
    print(f"  loss weights: regression={W_REGR}, binary={W_BIN}, mc={W_MC}")

    # ── Build Datasets ────────────────────────────────────────────────────
    print("\n[STEP 2] Building datasets (with scaler applied inside)...")

    print("  Loading train split...")
    df_train = pd.read_parquet(PROC_DIR/"state_train.parquet")
    df_train = df_train.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    train_ds = NetworkSequenceDataset(df_train, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=5)
    print(f"  Train sequences: {len(train_ds):,}")
    del df_train; gc.collect()

    print("  Loading val split...")
    df_val = pd.read_parquet(PROC_DIR/"state_val.parquet")
    df_val = df_val.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    val_ds = NetworkSequenceDataset(df_val, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=1)
    print(f"  Val sequences:   {len(val_ds):,}")
    del df_val; gc.collect()

    print("  Loading test split...")
    df_test = pd.read_parquet(PROC_DIR/"state_test.parquet")
    df_test = df_test.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    test_ds = NetworkSequenceDataset(df_test, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=1)
    print(f"  Test sequences:  {len(test_ds):,}")
    del df_test; gc.collect()

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,
                               num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False,
                               num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False,
                               num_workers=0, pin_memory=False)

    # ── Compute class weights from train ──────────────────────────────────
    print("\n[STEP 3] Computing class weights from training targets...")
    y_train_bin = train_ds.y_bin_arr[train_ds.indices + SEQ_LEN - 1]
    y_train_mc  = train_ds.y_mc_arr[train_ds.indices  + SEQ_LEN - 1]
    cw_bin = compute_class_weights(y_train_bin, 2,            device)
    cw_mc  = compute_class_weights(y_train_mc,  N_CLASSES_MC, device)
    print(f"  Binary class weights:     {cw_bin.cpu().numpy().round(4)}")
    print(f"  Multiclass weights (top5):{cw_mc.cpu().numpy()[:5].round(4)}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[STEP 4] Building GRU World Model...")
    model = GRUWorldModel(
        input_dim   = D,
        hidden_dim  = HIDDEN_DIM,
        num_layers  = NUM_LAYERS,
        n_classes_mc= N_CLASSES_MC,
        dropout     = DROPOUT
    ).to(device)
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture:\n{model}")

    # ── Loss & Optimizer ──────────────────────────────────────────────────
    criterion_state = nn.MSELoss()
    criterion_bin   = nn.CrossEntropyLoss(weight=cw_bin)
    criterion_mc    = nn.CrossEntropyLoss(weight=cw_mc)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, min_lr=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[STEP 5] Training (epochs={EPOCHS}, patience={PATIENCE})...")
    history = []
    best_val_f1   = -1.0
    best_state    = None
    patience_cnt  = 0
    best_epoch    = 0

    for epoch in range(1, EPOCHS + 1):
        t_ep = time.time()
        model.train()
        ep_loss = ep_ls = ep_lb = ep_lm = 0.0
        n_batches = 0

        for batch_idx, (X, y_state, y_bin, y_mc) in enumerate(train_loader):
            X       = X.to(device)
            y_state = y_state.to(device)
            y_bin   = y_bin.to(device)
            y_mc    = y_mc.to(device)

            optimizer.zero_grad()
            pred_state, pred_bin, pred_mc = model(X)

            l_s = criterion_state(pred_state, y_state)
            l_b = criterion_bin(pred_bin, y_bin)
            l_m = criterion_mc(pred_mc, y_mc)
            loss = W_REGR*l_s + W_BIN*l_b + W_MC*l_m

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item(); ep_ls += l_s.item()
            ep_lb   += l_b.item(); ep_lm += l_m.item()
            n_batches += 1

        # Epoch train metrics
        train_metrics = {
            "loss":            ep_loss/n_batches,
            "loss_regression": ep_ls/n_batches,
            "loss_binary":     ep_lb/n_batches,
            "loss_multiclass": ep_lm/n_batches,
        }

        # Validation
        val_metrics, _, _, _, _ = evaluate(
            model, val_loader, criterion_state, criterion_bin, criterion_mc, device, "Val")
        scheduler.step(val_metrics.get("bin_f1_macro", 0.0))

        ep_time = round(time.time()-t_ep, 1)
        val_f1  = val_metrics.get("bin_f1_macro", 0.0)
        val_auc = val_metrics.get("bin_roc_auc",  0.0)
        cur_lr  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_bin_f1={val_f1:.4f}  "
              f"val_auc={val_auc:.4f}  "
              f"val_mc_f1={val_metrics['mc_f1_macro']:.4f}  "
              f"lr={cur_lr:.2e}  {ep_time}s")

        history.append({"epoch":epoch,"train":train_metrics,"val":val_metrics,"lr":cur_lr})

        # Early stopping on val binary F1
        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_state   = copy.deepcopy(model.state_dict())
            best_epoch   = epoch
            patience_cnt = 0
            print(f"    ✓ New best val binary F1: {best_val_f1:.4f} — saved checkpoint")
        else:
            patience_cnt += 1
            print(f"    Patience: {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    print(f"\n  Training complete. Best epoch: {best_epoch}  Best val F1: {best_val_f1:.4f}")

    # ── Load best model ───────────────────────────────────────────────────
    model.load_state_dict(best_state)

    # ── Final evaluation — VAL ────────────────────────────────────────────
    print("\n[STEP 6] Final evaluation on VAL set (best checkpoint)...")
    val_metrics_final, y_val_bin, p_val_bin, y_val_mc, p_val_mc = evaluate(
        model, val_loader, criterion_state, criterion_bin, criterion_mc, device, "Val")
    print(f"\n  VAL Binary:")
    for k,v in val_metrics_final.items():
        if "bin" in k: print(f"    {k:<28}: {v:.4f}")
    print(f"\n  VAL Multiclass:")
    for k,v in val_metrics_final.items():
        if "mc" in k: print(f"    {k:<28}: {v:.4f}")

    # ── Final evaluation — TEST ───────────────────────────────────────────
    print("\n[STEP 7] Final evaluation on TEST set...")
    test_metrics_final, y_test_bin, p_test_bin, y_test_mc, p_test_mc = evaluate(
        model, test_loader, criterion_state, criterion_bin, criterion_mc, device, "Test")
    print(f"\n  TEST Binary:")
    for k,v in test_metrics_final.items():
        if "bin" in k: print(f"    {k:<28}: {v:.4f}")
    print(f"\n  TEST Multiclass:")
    for k,v in test_metrics_final.items():
        if "mc" in k: print(f"    {k:<28}: {v:.4f}")

    print(f"\n  Binary Classification Report (Test):")
    print(classification_report(y_test_bin, p_test_bin,
                                 target_names=["BENIGN","ATTACK"], zero_division=0))
    print(f"  Multiclass Classification Report (Test):")
    idx2class = label_enc["idx2class"]
    p_names   = [idx2class[str(p)] for p in p_test_mc]
    t_names   = [idx2class[str(t)] for t in y_test_mc]
    present   = sorted(set(t_names)|set(p_names))
    print(classification_report(t_names, p_names, labels=present, zero_division=0))

    # Per-class F1
    per_f1_bin = f1_score(y_test_bin, p_test_bin, average=None, zero_division=0)
    per_f1_mc  = f1_score(y_test_mc,  p_test_mc,  average=None, zero_division=0)
    labels_mc  = sorted(set(y_test_mc))
    print(f"\n  Per-class F1 (Test — Multiclass):")
    for cls_idx, f1v in zip(labels_mc, per_f1_mc):
        cnt = int(np.sum(y_test_mc == cls_idx))
        print(f"    {CLASS_NAMES[cls_idx]:<35}: F1={f1v:.4f}  (support={cnt})")

    # ── Save model ────────────────────────────────────────────────────────
    print("\n[STEP 8] Saving model...")
    model_path = WM_DIR / "gru_world_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": {
            "input_dim":D,"hidden_dim":HIDDEN_DIM,"num_layers":NUM_LAYERS,
            "n_classes_mc":N_CLASSES_MC,"dropout":DROPOUT,"seq_len":SEQ_LEN
        },
        "best_epoch":  best_epoch,
        "best_val_f1": best_val_f1,
        "feature_cols": FEATURE_COLS,
    }, model_path)
    print(f"  Model saved: {model_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[STEP 9] Saving plots...")

    # Training curves
    plot_training_curves(history, PLOTS_DIR/"phase8_gru_training_curves.png")
    print("  Training curves saved.")

    # Confusion matrix — binary test
    cm_bin = confusion_matrix(y_test_bin, p_test_bin, labels=[0,1])
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues",
                xticklabels=["BENIGN","ATTACK"], yticklabels=["BENIGN","ATTACK"], ax=ax)
    ax.set_title("GRU World Model — Binary CM (Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"phase8_gru_cm_binary.png", dpi=150, bbox_inches="tight"); plt.close()

    # Confusion matrix — multiclass test
    present_int = sorted(set(y_test_mc))
    present_names = [CLASS_NAMES[i] for i in present_int]
    cm_mc = confusion_matrix(y_test_mc, p_test_mc, labels=present_int)
    fig, ax = plt.subplots(figsize=(8,7))
    sns.heatmap(cm_mc, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names, ax=ax, linewidths=0.4)
    ax.set_title("GRU World Model — Multiclass CM (Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=9); plt.tight_layout()
    plt.savefig(PLOTS_DIR/"phase8_gru_cm_multiclass.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Confusion matrices saved.")

    # ── Comparison table: LR vs XGB vs GRU ───────────────────────────────
    print("\n[STEP 10] Full model comparison (LR vs XGB vs GRU — Test):")
    with open(METRICS_DIR/"phase6_lr_metrics.json")  as f: lr_m  = json.load(f)
    with open(METRICS_DIR/"phase7_xgb_metrics.json") as f: xgb_m = json.load(f)

    print(f"\n  BINARY (BENIGN vs ATTACK) — Test:")
    print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10} {'GRU':>10}")
    print(f"    {'-'*57}")
    for metric, gru_key in [("accuracy","bin_accuracy"),("f1_macro","bin_f1_macro"),
                              ("recall_macro","bin_recall_macro"),("roc_auc","bin_roc_auc")]:
        lv  = lr_m["binary"]["test"].get(metric, 0)
        xv  = xgb_m["binary"]["test"].get(metric, 0)
        gv  = test_metrics_final.get(gru_key, 0)
        best = max(lv, xv, gv)
        def fmt(v): return f"**{v:.4f}**" if v==best else f"{v:.4f}"
        print(f"    {metric:<25} {lv:>10.4f} {xv:>10.4f} {gv:>10.4f}{'  ← BEST GRU' if gv==best else ''}")

    print(f"\n  MULTICLASS — Test:")
    print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10} {'GRU':>10}")
    print(f"    {'-'*57}")
    for metric, gru_key in [("accuracy","mc_accuracy"),("f1_macro","mc_f1_macro"),
                              ("recall_macro","mc_recall_macro")]:
        lv  = lr_m["multiclass"]["test"].get(metric, 0)
        xv  = xgb_m["multiclass"]["test"].get(metric, 0)
        gv  = test_metrics_final.get(gru_key, 0)
        print(f"    {metric:<25} {lv:>10.4f} {xv:>10.4f} {gv:>10.4f}{'  ← BEST GRU' if gv==max(lv,xv,gv) else ''}")

    # ── Save final metrics ────────────────────────────────────────────────
    elapsed = round(time.time()-t0, 2)
    per_f1_test_mc = {CLASS_NAMES[i]: round(float(v),4) for i,v in zip(labels_mc, per_f1_mc)}
    final_metrics = {
        "phase": 8, "model": "GRU_WorldModel",
        "architecture": {
            "input_dim":D,"hidden_dim":HIDDEN_DIM,"num_layers":NUM_LAYERS,
            "n_classes_mc":N_CLASSES_MC,"dropout":DROPOUT,"seq_len":SEQ_LEN,
            "n_params":n_params
        },
        "training": {
            "epochs_run": len(history), "best_epoch": best_epoch,
            "best_val_bin_f1": round(best_val_f1, 6),
            "loss_weights": {"regression":W_REGR,"binary":W_BIN,"multiclass":W_MC},
            "optimizer":"Adam","lr":LR,"weight_decay":WD,
            "scheduler":"ReduceLROnPlateau"
        },
        "val":  {k: round(float(v),6) for k,v in val_metrics_final.items()},
        "test": {k: round(float(v),6) for k,v in test_metrics_final.items()},
        "test_per_class_f1_mc": per_f1_test_mc,
        "history": [{
            "epoch": h["epoch"],
            "train_loss": round(h["train"]["loss"],6),
            "val_loss":   round(h["val"]["loss"],6),
            "val_bin_f1": round(h["val"].get("bin_f1_macro",0),6),
            "val_auc":    round(h["val"].get("bin_roc_auc",0),6),
            "val_mc_f1":  round(h["val"].get("mc_f1_macro",0),6),
        } for h in history],
        "elapsed_seconds": elapsed
    }
    mpath = METRICS_DIR/"phase8_gru_metrics.json"
    with open(mpath,"w") as f: json.dump(final_metrics, f, indent=2)
    print(f"\n  Metrics saved: {mpath}")

    print(f"\n{'='*72}")
    print(f"PHASE 8 COMPLETE | GRU WORLD MODEL | Elapsed: {elapsed}s")
    print(f"  Params:         {n_params:,}")
    print(f"  Best epoch:     {best_epoch}")
    print(f"  TEST Binary  — accuracy={test_metrics_final['bin_accuracy']:.4f}  "
          f"f1={test_metrics_final['bin_f1_macro']:.4f}  "
          f"auc={test_metrics_final.get('bin_roc_auc',0):.4f}")
    print(f"  TEST MC      — accuracy={test_metrics_final['mc_accuracy']:.4f}  "
          f"f1={test_metrics_final['mc_f1_macro']:.4f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
