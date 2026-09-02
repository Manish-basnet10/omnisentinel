"""
PHASE 9 – LSTM World Model (with Temporal Attention)
=====================================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

IMPROVEMENTS OVER PHASE 8 GRU:
  1. LSTM (hidden + cell state) for richer temporal memory
  2. Temporal self-attention over all L timesteps instead of last-step only
  3. Early stopping on val ATTACK recall (class 1) — more stable than F1-macro
     under extreme val imbalance (98.9% BENIGN in val)
  4. Hidden dim = 256 (GRU used 128)
  5. Binary loss weight = 0.50 (GRU used 0.40) — prioritise attack detection
  6. Gradient accumulation steps = 2 for effective batch = 2048

ARCHITECTURE
------------
  LSTM(input=60, hidden=256, layers=2, dropout=0.3, batch_first=True)
  → Temporal Attention: score = softmax(W_a · h_t) weighted sum over all t
  → context vector (batch, 256)
  → LayerNorm(256)
  → Head 1: Linear(256→128)→GELU→Dropout→Linear(128→60)   [regression]
  → Head 2: Linear(256→64)→GELU→Dropout→Linear(64→2)      [binary]
  → Head 3: Linear(256→64)→GELU→Dropout→Linear(64→15)     [multiclass]

TOTAL LOSS = 0.15*L_regression + 0.50*L_binary + 0.35*L_multiclass

STRICT RULES:
  - Same scaler as Phase 6 (standard_scaler.pkl) — NO re-fitting
  - Same train/val/test splits as all previous phases
  - All metrics from real predictions — ZERO fabrication
  - Fixed seed = 42
"""

import os, json, warnings, time, gc, pickle, copy
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
)

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"
WM_DIR      = BASE_DIR / "models" / "world_model"
WM_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN      = 10
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_TRAIN  = 1024
BATCH_EVAL   = 2048
GRAD_ACCUM   = 2          # effective batch = 2048
LR           = 1e-3
WD           = 1e-5
EPOCHS       = 20
PATIENCE     = 5          # on val ATTACK recall
W_REGR       = 0.15
W_BIN        = 0.50
W_MC         = 0.35
N_CLASSES_MC = 15
ATTACK_CLASS = 1          # binary: 1=ATTACK, used for early stopping


# ══════════════════════════════════════════════════════════════════════════════
# DATASET  (same as Phase 8 — reused verbatim)
# ══════════════════════════════════════════════════════════════════════════════
class NetworkSequenceDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, scaler=None, stride=1):
        self.seq_len   = seq_len
        self.next_cols = [f"next_{c}" for c in feature_cols]
        self.D         = len(feature_cols)

        feat_arr      = df[feature_cols].values.astype(np.float32)
        next_feat_arr = df[self.next_cols].values.astype(np.float32)
        if scaler is not None:
            feat_arr      = scaler.transform(feat_arr).astype(np.float32)
            next_feat_arr = scaler.transform(next_feat_arr).astype(np.float32)

        self.feat_arr      = feat_arr
        self.next_feat_arr = next_feat_arr
        self.y_bin_arr     = df["next_label_binary"].values.astype(np.int64)
        self.y_mc_arr      = df["next_label_multiclass"].values.astype(np.int64)

        day_orders  = df["day_order"].values
        is_boundary = df["is_boundary"].values
        n = len(df)
        valid_starts = []
        i = 0
        while i + seq_len < n:
            end = i + seq_len
            if day_orders[i] != day_orders[end]: i += 1; continue
            if np.any(is_boundary[i:end]):        i += 1; continue
            valid_starts.append(i)
            i += stride
        self.indices = np.array(valid_starts, dtype=np.int64)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]; end = start + self.seq_len
        X       = torch.from_numpy(self.feat_arr[start:end].copy())
        y_state = torch.from_numpy(self.next_feat_arr[end-1].copy())
        y_bin   = torch.tensor(int(self.y_bin_arr[end-1]), dtype=torch.long)
        y_mc    = torch.tensor(int(self.y_mc_arr[end-1]),  dtype=torch.long)
        return X, y_state, y_bin, y_mc


# ══════════════════════════════════════════════════════════════════════════════
# TEMPORAL ATTENTION MODULE
# ══════════════════════════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over LSTM outputs.
    Learns which timesteps matter most for prediction.

    Input:  lstm_out  (batch, seq_len, hidden)
    Output: context   (batch, hidden)
            attn_w    (batch, seq_len)  — interpretable weights
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_a  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_a  = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq_len, hidden)
        energy   = torch.tanh(self.W_a(lstm_out))   # (batch, seq_len, hidden)
        scores   = self.v_a(energy).squeeze(-1)      # (batch, seq_len)
        attn_w   = F.softmax(scores, dim=-1)         # (batch, seq_len)
        context  = torch.bmm(attn_w.unsqueeze(1), lstm_out).squeeze(1)  # (batch, hidden)
        return context, attn_w


# ══════════════════════════════════════════════════════════════════════════════
# LSTM WORLD MODEL
# ══════════════════════════════════════════════════════════════════════════════
class LSTMWorldModel(nn.Module):
    """
    Multi-task LSTM World Model with Temporal Attention.
    Three prediction heads: state regression, binary cls, multiclass cls.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes_mc, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )
        self.attention  = TemporalAttention(hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Head 1: next-state regression
        self.head_state = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim)
        )
        # Head 2: binary classification
        self.head_binary = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
        # Head 3: multiclass classification
        self.head_mc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes_mc)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden)
        context, attn_w = self.attention(lstm_out)  # (batch, hidden), (batch, seq_len)
        context = self.layer_norm(context)
        return self.head_state(context), self.head_binary(context), self.head_mc(context), attn_w

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_class_weights(y_array, n_classes, device):
    counts  = np.bincount(y_array, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def evaluate(model, loader, criterion_state, criterion_bin, criterion_mc, device):
    model.eval()
    all_y_bin, all_p_bin, all_prob_bin = [], [], []
    all_y_mc,  all_p_mc               = [], []
    total_loss = total_ls = total_lb = total_lm = 0.0
    n_batches  = 0

    with torch.no_grad():
        for X, y_state, y_bin, y_mc in loader:
            X = X.to(device); y_state = y_state.to(device)
            y_bin = y_bin.to(device); y_mc = y_mc.to(device)

            pred_state, pred_bin, pred_mc, _ = model(X)
            l_s  = criterion_state(pred_state, y_state)
            l_b  = criterion_bin(pred_bin, y_bin)
            l_m  = criterion_mc(pred_mc, y_mc)
            loss = W_REGR*l_s + W_BIN*l_b + W_MC*l_m

            total_loss += loss.item(); total_ls += l_s.item()
            total_lb   += l_b.item(); total_lm += l_m.item()
            n_batches  += 1

            all_y_bin.extend(y_bin.cpu().numpy())
            all_p_bin.extend(pred_bin.argmax(1).cpu().numpy())
            all_prob_bin.extend(F.softmax(pred_bin,dim=1)[:,1].cpu().numpy())
            all_y_mc.extend(y_mc.cpu().numpy())
            all_p_mc.extend(pred_mc.argmax(1).cpu().numpy())

    all_y_bin = np.array(all_y_bin); all_p_bin = np.array(all_p_bin)
    all_prob_bin = np.array(all_prob_bin)
    all_y_mc = np.array(all_y_mc); all_p_mc = np.array(all_p_mc)

    # ATTACK recall (class 1) — early stopping signal
    attack_mask = (all_y_bin == 1)
    if attack_mask.sum() > 0:
        attack_recall = float(np.mean(all_p_bin[attack_mask] == 1))
    else:
        attack_recall = 0.0

    metrics = {
        "loss": total_loss/n_batches, "loss_regression": total_ls/n_batches,
        "loss_binary": total_lb/n_batches, "loss_multiclass": total_lm/n_batches,
        "bin_accuracy":       float(accuracy_score(all_y_bin, all_p_bin)),
        "bin_f1_macro":       float(f1_score(all_y_bin, all_p_bin, average="macro",    zero_division=0)),
        "bin_f1_weighted":    float(f1_score(all_y_bin, all_p_bin, average="weighted", zero_division=0)),
        "bin_recall_macro":   float(recall_score(all_y_bin, all_p_bin, average="macro", zero_division=0)),
        "bin_precision_macro":float(precision_score(all_y_bin, all_p_bin, average="macro", zero_division=0)),
        "attack_recall":      attack_recall,      # ← KEY: early stopping signal
        "mc_accuracy":        float(accuracy_score(all_y_mc, all_p_mc)),
        "mc_f1_macro":        float(f1_score(all_y_mc, all_p_mc, average="macro",    zero_division=0)),
        "mc_f1_weighted":     float(f1_score(all_y_mc, all_p_mc, average="weighted", zero_division=0)),
        "mc_recall_macro":    float(recall_score(all_y_mc, all_p_mc, average="macro", zero_division=0)),
    }
    if len(np.unique(all_y_bin)) == 2:
        metrics["bin_roc_auc"]       = float(roc_auc_score(all_y_bin, all_prob_bin))
        metrics["bin_avg_precision"] = float(average_precision_score(all_y_bin, all_prob_bin))

    return metrics, all_y_bin, all_p_bin, all_y_mc, all_p_mc


def plot_training_curves(history, path):
    metrics_to_plot = [
        ("loss",          "Total Loss",        "#e74c3c"),
        ("loss_binary",   "Binary CE Loss",    "#e67e22"),
        ("attack_recall", "Val ATTACK Recall", "#c0392b"),   # ← new stopping metric
        ("bin_f1_macro",  "Binary F1-macro",   "#27ae60"),
        ("bin_roc_auc",   "Binary ROC-AUC",    "#2980b9"),
        ("mc_f1_macro",   "MC F1-macro",       "#16a085"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title, color) in zip(axes.flatten(), metrics_to_plot):
        train_vals = [h["train"].get(key) for h in history]
        val_vals   = [h["val"].get(key)   for h in history]
        xs = list(range(1, len(history)+1))
        if any(v is not None for v in train_vals):
            ax.plot(xs, [v or 0 for v in train_vals], label="Train", color=color, lw=2, alpha=0.7)
        if any(v is not None for v in val_vals):
            ax.plot(xs, [v or 0 for v in val_vals],   label="Val",   color=color, lw=2, ls="--")
        ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_xlabel("Epoch")
        ax.legend(fontsize=8); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.suptitle("LSTM World Model — Training History", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 9 – LSTM WORLD MODEL (TEMPORAL ATTENTION, MULTI-TASK)")
    print("="*72)

    # ── Device ────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"\n  Device: {device}")
    print(f"  Improvements over GRU: LSTM+Attention, hidden=256, early-stop=ATTACK recall")

    # ── Load definitions ──────────────────────────────────────────────────
    print("\n[STEP 1] Loading definitions...")
    with open(PROC_DIR/"feature_list.json")   as f: feat_def  = json.load(f)
    with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)
    with open(MODELS_DIR/"standard_scaler.pkl","rb") as f:
        scaler = pickle.load(f)["scaler"]

    FEATURE_COLS = feat_def["all_numeric_features"]
    CLASS_NAMES  = label_enc["classes"]
    D            = len(FEATURE_COLS)
    print(f"  D={D}, seq_len={SEQ_LEN}, classes={N_CLASSES_MC}")
    print(f"  hidden={HIDDEN_DIM}, layers={NUM_LAYERS}, dropout={DROPOUT}")
    print(f"  loss: regr={W_REGR}, bin={W_BIN}, mc={W_MC}")
    print(f"  early stop: val ATTACK recall (class 1)")

    # ── Build Datasets ────────────────────────────────────────────────────
    print("\n[STEP 2] Building datasets...")
    df_train = pd.read_parquet(PROC_DIR/"state_train.parquet")
    df_train = df_train.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    train_ds = NetworkSequenceDataset(df_train, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=5)
    print(f"  Train sequences: {len(train_ds):,}")
    del df_train; gc.collect()

    df_val = pd.read_parquet(PROC_DIR/"state_val.parquet")
    df_val = df_val.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    val_ds = NetworkSequenceDataset(df_val, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=1)
    print(f"  Val sequences:   {len(val_ds):,}")
    del df_val; gc.collect()

    df_test = pd.read_parquet(PROC_DIR/"state_test.parquet")
    df_test = df_test.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    test_ds = NetworkSequenceDataset(df_test, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=1)
    print(f"  Test sequences:  {len(test_ds):,}")
    del df_test; gc.collect()

    train_loader = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False, num_workers=0)

    # ── Class weights ─────────────────────────────────────────────────────
    print("\n[STEP 3] Computing class weights...")
    y_train_bin = train_ds.y_bin_arr[train_ds.indices + SEQ_LEN - 1]
    y_train_mc  = train_ds.y_mc_arr[train_ds.indices  + SEQ_LEN - 1]
    cw_bin = compute_class_weights(y_train_bin, 2,            device)
    cw_mc  = compute_class_weights(y_train_mc,  N_CLASSES_MC, device)
    print(f"  Binary weights:     {cw_bin.cpu().numpy().round(4)}")
    print(f"  ATTACK weight:      {cw_bin[1].item():.4f}  (upweighted {cw_bin[1].item()/cw_bin[0].item():.1f}x)")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[STEP 4] Building LSTM World Model...")
    model = LSTMWorldModel(
        input_dim    = D,
        hidden_dim   = HIDDEN_DIM,
        num_layers   = NUM_LAYERS,
        n_classes_mc = N_CLASSES_MC,
        dropout      = DROPOUT
    ).to(device)
    n_params = model.count_params()
    print(f"  Parameters: {n_params:,}")
    print(f"  Architecture:\n{model}")

    # ── Loss & Optimizer ──────────────────────────────────────────────────
    criterion_state = nn.MSELoss()
    criterion_bin   = nn.CrossEntropyLoss(weight=cw_bin)
    criterion_mc    = nn.CrossEntropyLoss(weight=cw_mc)
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, min_lr=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[STEP 5] Training (epochs={EPOCHS}, patience={PATIENCE} on ATTACK recall)...")
    history = []
    best_attack_recall = -1.0
    best_state         = None
    patience_cnt       = 0
    best_epoch         = 0

    for epoch in range(1, EPOCHS+1):
        t_ep = time.time()
        model.train()
        ep_loss = ep_ls = ep_lb = ep_lm = 0.0
        n_batches = 0
        optimizer.zero_grad()   # gradient accumulation

        for batch_idx, (X, y_state, y_bin, y_mc) in enumerate(train_loader):
            X = X.to(device); y_state = y_state.to(device)
            y_bin = y_bin.to(device); y_mc = y_mc.to(device)

            pred_state, pred_bin, pred_mc, _ = model(X)
            l_s  = criterion_state(pred_state, y_state)
            l_b  = criterion_bin(pred_bin, y_bin)
            l_m  = criterion_mc(pred_mc, y_mc)
            loss = (W_REGR*l_s + W_BIN*l_b + W_MC*l_m) / GRAD_ACCUM
            loss.backward()

            ep_loss += (W_REGR*l_s + W_BIN*l_b + W_MC*l_m).item()
            ep_ls   += l_s.item(); ep_lb += l_b.item(); ep_lm += l_m.item()
            n_batches += 1

            if (batch_idx+1) % GRAD_ACCUM == 0 or (batch_idx+1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        train_metrics = {
            "loss":            ep_loss/n_batches,
            "loss_regression": ep_ls/n_batches,
            "loss_binary":     ep_lb/n_batches,
            "loss_multiclass": ep_lm/n_batches,
        }

        # Validation
        val_metrics, _, _, _, _ = evaluate(
            model, val_loader, criterion_state, criterion_bin, criterion_mc, device)
        scheduler.step(val_metrics.get("attack_recall", 0.0))

        ep_time   = round(time.time()-t_ep, 1)
        atk_rec   = val_metrics.get("attack_recall", 0.0)
        val_f1    = val_metrics.get("bin_f1_macro",  0.0)
        val_auc   = val_metrics.get("bin_roc_auc",   0.0)
        val_mc_f1 = val_metrics.get("mc_f1_macro",   0.0)
        cur_lr    = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"train_loss={train_metrics['loss']:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_atk_recall={atk_rec:.4f}  "
              f"val_bin_f1={val_f1:.4f}  "
              f"val_auc={val_auc:.4f}  "
              f"val_mc_f1={val_mc_f1:.4f}  "
              f"lr={cur_lr:.2e}  {ep_time}s")

        history.append({"epoch":epoch,"train":train_metrics,"val":val_metrics,"lr":cur_lr})

        # Early stopping on val ATTACK recall
        if atk_rec > best_attack_recall:
            best_attack_recall = atk_rec
            best_state         = copy.deepcopy(model.state_dict())
            best_epoch         = epoch
            patience_cnt       = 0
            print(f"    ✓ Best val ATTACK recall: {best_attack_recall:.4f} — saved checkpoint")
        else:
            patience_cnt += 1
            print(f"    Patience: {patience_cnt}/{PATIENCE}")
            if patience_cnt >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\n  Training done. Best epoch: {best_epoch}  Best val ATTACK recall: {best_attack_recall:.4f}")

    # ── Load best & evaluate ──────────────────────────────────────────────
    model.load_state_dict(best_state)

    print("\n[STEP 6] Final evaluation — VAL set...")
    val_m, y_val_bin, p_val_bin, y_val_mc, p_val_mc = evaluate(
        model, val_loader, criterion_state, criterion_bin, criterion_mc, device)
    print(f"\n  VAL Binary:")
    for k,v in val_m.items():
        if "bin" in k or k=="attack_recall":
            print(f"    {k:<28}: {v:.4f}")
    print(f"\n  VAL Multiclass:")
    for k,v in val_m.items():
        if "mc" in k: print(f"    {k:<28}: {v:.4f}")

    print("\n[STEP 7] Final evaluation — TEST set...")
    test_m, y_test_bin, p_test_bin, y_test_mc, p_test_mc = evaluate(
        model, test_loader, criterion_state, criterion_bin, criterion_mc, device)
    print(f"\n  TEST Binary:")
    for k,v in test_m.items():
        if "bin" in k or k=="attack_recall":
            print(f"    {k:<28}: {v:.4f}")
    print(f"\n  TEST Multiclass:")
    for k,v in test_m.items():
        if "mc" in k: print(f"    {k:<28}: {v:.4f}")

    print(f"\n  Binary Classification Report (Test):")
    print(classification_report(y_test_bin, p_test_bin,
                                 target_names=["BENIGN","ATTACK"], zero_division=0))

    idx2class = label_enc["idx2class"]
    p_names   = [idx2class[str(p)] for p in p_test_mc]
    t_names   = [idx2class[str(t)] for t in y_test_mc]
    present   = sorted(set(t_names)|set(p_names))
    print(f"  Multiclass Classification Report (Test):")
    print(classification_report(t_names, p_names, labels=present, zero_division=0))

    # Per-class F1 test
    labels_mc   = sorted(set(y_test_mc))
    per_f1_mc   = f1_score(y_test_mc, p_test_mc, average=None, zero_division=0)
    per_f1_test = {CLASS_NAMES[i]: round(float(v),4) for i,v in zip(labels_mc, per_f1_mc)}
    print(f"\n  Per-class F1 (Test — Multiclass):")
    for cls,f1v in per_f1_test.items():
        cnt = int(np.sum(y_test_mc == label_enc["class2idx"].get(cls,-1)))
        print(f"    {cls:<35}: F1={f1v:.4f}  (support={cnt})")

    # ── Save model ────────────────────────────────────────────────────────
    print("\n[STEP 8] Saving LSTM model...")
    model_path = WM_DIR/"lstm_world_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hyperparams": {"input_dim":D,"hidden_dim":HIDDEN_DIM,"num_layers":NUM_LAYERS,
                        "n_classes_mc":N_CLASSES_MC,"dropout":DROPOUT,"seq_len":SEQ_LEN},
        "best_epoch":  best_epoch,
        "best_val_attack_recall": best_attack_recall,
        "feature_cols": FEATURE_COLS,
    }, model_path)
    print(f"  Saved: {model_path}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[STEP 9] Saving plots...")
    plot_training_curves(history, PLOTS_DIR/"phase9_lstm_training_curves.png")

    for (y_t, p_t, labels, lnames, title, fname) in [
        (y_test_bin, p_test_bin, [0,1], ["BENIGN","ATTACK"],
         "LSTM World Model — Binary CM (Test)", "phase9_lstm_cm_binary.png"),
        (y_test_mc, p_test_mc, sorted(set(y_test_mc)),
         [CLASS_NAMES[i] for i in sorted(set(y_test_mc))],
         "LSTM World Model — Multiclass CM (Test)", "phase9_lstm_cm_multiclass.png"),
    ]:
        cm = confusion_matrix(y_t, p_t, labels=labels)
        fig, ax = plt.subplots(figsize=(6,5) if len(labels)==2 else (8,7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=lnames, yticklabels=lnames, ax=ax, linewidths=0.4)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        if len(labels)>2: plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR/fname, dpi=150, bbox_inches="tight"); plt.close()
    print("  Plots saved.")

    # ── Attention weight visualization (sample batch) ─────────────────────
    print("\n[STEP 10] Visualizing temporal attention weights...")
    model.eval()
    sample_X, sample_y_state, sample_y_bin, sample_y_mc = next(iter(val_loader))
    with torch.no_grad():
        _, _, _, attn_w = model(sample_X.to(device))
    attn_w_np = attn_w.cpu().numpy()[:32]  # first 32 samples
    fig, ax = plt.subplots(figsize=(12,5))
    sns.heatmap(attn_w_np, cmap="YlOrRd", ax=ax, yticklabels=False,
                xticklabels=[f"t-{SEQ_LEN-1-i}" for i in range(SEQ_LEN)])
    ax.set_xlabel("Timestep (t-9 = oldest, t-0 = most recent)", fontsize=10)
    ax.set_ylabel("Sample sequences (32)", fontsize=10)
    ax.set_title("LSTM Temporal Attention Weights (Val batch)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR/"phase9_lstm_attention_weights.png", dpi=150, bbox_inches="tight"); plt.close()
    print("  Attention weight map saved.")

    # ── Full comparison: LR vs XGB vs GRU vs LSTM ────────────────────────
    print("\n[STEP 11] Full comparison — LR vs XGB vs GRU vs LSTM (Test):")
    with open(METRICS_DIR/"phase6_lr_metrics.json")  as f: lr_m  = json.load(f)
    with open(METRICS_DIR/"phase7_xgb_metrics.json") as f: xgb_m = json.load(f)
    with open(METRICS_DIR/"phase8_gru_metrics.json") as f: gru_m = json.load(f)

    print(f"\n  BINARY — Test:")
    print(f"    {'Metric':<23} {'LR':>9} {'XGB':>9} {'GRU':>9} {'LSTM':>9}  Best")
    print(f"    {'-'*67}")
    binary_rows = [
        ("accuracy",     "bin_accuracy"),
        ("f1_macro",     "bin_f1_macro"),
        ("f1_weighted",  "bin_f1_weighted"),
        ("recall_macro", "bin_recall_macro"),
        ("roc_auc",      "bin_roc_auc"),
    ]
    for m_key, g_key in binary_rows:
        lv  = lr_m["binary"]["test"].get(m_key, 0)
        xv  = xgb_m["binary"]["test"].get(m_key, 0)
        gv  = gru_m["test"].get(g_key, 0)
        lstv= test_m.get(g_key, 0)
        vals= {"LR":lv,"XGB":xv,"GRU":gv,"LSTM":lstv}
        best= max(vals, key=vals.get)
        print(f"    {m_key:<23} {lv:>9.4f} {xv:>9.4f} {gv:>9.4f} {lstv:>9.4f}  ← {best}")

    print(f"\n  MULTICLASS — Test:")
    print(f"    {'Metric':<23} {'LR':>9} {'XGB':>9} {'GRU':>9} {'LSTM':>9}  Best")
    print(f"    {'-'*67}")
    mc_rows = [
        ("accuracy",     "mc_accuracy"),
        ("f1_macro",     "mc_f1_macro"),
        ("recall_macro", "mc_recall_macro"),
    ]
    for m_key, g_key in mc_rows:
        lv  = lr_m["multiclass"]["test"].get(m_key, 0)
        xv  = xgb_m["multiclass"]["test"].get(m_key, 0)
        gv  = gru_m["test"].get(g_key, 0)
        lstv= test_m.get(g_key, 0)
        vals= {"LR":lv,"XGB":xv,"GRU":gv,"LSTM":lstv}
        best= max(vals, key=vals.get)
        print(f"    {m_key:<23} {lv:>9.4f} {xv:>9.4f} {gv:>9.4f} {lstv:>9.4f}  ← {best}")

    # ── Save metrics ──────────────────────────────────────────────────────
    elapsed = round(time.time()-t0, 2)
    final_metrics = {
        "phase": 9, "model": "LSTM_WorldModel_Attention", "seed": SEED,
        "architecture": {
            "input_dim":D,"hidden_dim":HIDDEN_DIM,"num_layers":NUM_LAYERS,
            "n_classes_mc":N_CLASSES_MC,"dropout":DROPOUT,"seq_len":SEQ_LEN,
            "attention":"TemporalAttention (additive)","n_params":n_params
        },
        "training": {
            "epochs_run":len(history),"best_epoch":best_epoch,
            "best_val_attack_recall":round(best_attack_recall,6),
            "early_stop_criterion":"val_attack_recall",
            "loss_weights":{"regression":W_REGR,"binary":W_BIN,"multiclass":W_MC},
            "optimizer":"Adam","lr":LR,"weight_decay":WD,"grad_accum":GRAD_ACCUM
        },
        "val":  {k:round(float(v),6) for k,v in val_m.items()},
        "test": {k:round(float(v),6) for k,v in test_m.items()},
        "test_per_class_f1_mc": per_f1_test,
        "history": [{"epoch":h["epoch"],
                     "train_loss":round(h["train"]["loss"],6),
                     "val_loss":round(h["val"]["loss"],6),
                     "val_attack_recall":round(h["val"].get("attack_recall",0),6),
                     "val_bin_f1":round(h["val"].get("bin_f1_macro",0),6),
                     "val_auc":round(h["val"].get("bin_roc_auc",0),6),
                     "val_mc_f1":round(h["val"].get("mc_f1_macro",0),6)} for h in history],
        "elapsed_seconds": elapsed
    }
    mpath = METRICS_DIR/"phase9_lstm_metrics.json"
    with open(mpath,"w") as f: json.dump(final_metrics, f, indent=2)
    print(f"\n  Metrics saved: {mpath}")

    print(f"\n{'='*72}")
    print(f"PHASE 9 COMPLETE | LSTM WORLD MODEL | Elapsed: {elapsed}s")
    print(f"  Params:       {n_params:,}")
    print(f"  Best epoch:   {best_epoch}")
    print(f"  TEST Binary — acc={test_m['bin_accuracy']:.4f}  "
          f"f1={test_m['bin_f1_macro']:.4f}  "
          f"auc={test_m.get('bin_roc_auc',0):.4f}  "
          f"atk_recall={test_m['attack_recall']:.4f}")
    print(f"  TEST MC     — acc={test_m['mc_accuracy']:.4f}  "
          f"f1={test_m['mc_f1_macro']:.4f}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
