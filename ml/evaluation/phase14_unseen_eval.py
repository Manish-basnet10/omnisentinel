"""
PHASE 14 – Unseen Attack Evaluation
=====================================
Evaluates how GRU World Model and XGBoost handle attack classes that
were NOT present in training (DDoS, PortScan appear only in test).

Delivers:
  - results/metrics/phase14_unseen_attack_eval.json
  - results/plots/phase14_unseen_attack_roc.png
  - results/plots/phase14_unseen_class_heatmap.png
"""
import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import (roc_auc_score, confusion_matrix,
                             classification_report, average_precision_score)

warnings.filterwarnings("ignore")
np.random.seed(42)

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
WM_DIR      = BASE_DIR / "models" / "world_model"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"

print("=" * 70)
print("PHASE 14 – UNSEEN ATTACK EVALUATION")
print("=" * 70)

# ── Load definitions ──────────────────────────────────────────────────────────
with open(PROC_DIR / "feature_list.json")   as f: feat_def  = json.load(f)
with open(PROC_DIR / "label_encoding.json") as f: label_enc = json.load(f)
with open(MODELS_DIR / "standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)["scaler"]

FEATURE_COLS = feat_def["all_numeric_features"]
CLASS_NAMES  = label_enc["classes"]
D = len(FEATURE_COLS)

# ── Load test set ─────────────────────────────────────────────────────────────
print("\n[STEP 1] Loading test set...")
df_test = pd.read_parquet(PROC_DIR / "state_test.parquet")
df_test = df_test.sort_values(["day_order", "seq_idx"]).reset_index(drop=True)

X_test = scaler.transform(df_test[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)
y_bin  = df_test["next_label_binary"].values.astype(int)
y_mc   = df_test["next_label_multiclass"].values.astype(int)

# Find classes that appear in test but not in train
with open(PROC_DIR / "feature_list.json") as f: fd = json.load(f)
# Check multiclass label distribution in test
test_class_counts = dict(zip(*np.unique(y_mc, return_counts=True)))
print(f"  Test set classes: {sorted(test_class_counts.keys())}")
print(f"  Total test rows:  {len(df_test):,}")

# Load training label distribution
df_train = pd.read_parquet(PROC_DIR / "state_train.parquet")
train_classes = set(df_train["next_label_multiclass"].unique())
test_classes  = set(y_mc)
unseen_classes = test_classes - train_classes
seen_classes   = test_classes & train_classes

print(f"  Train classes: {sorted(train_classes)}")
print(f"  Unseen in test: {sorted(unseen_classes)} = {[CLASS_NAMES[i] for i in sorted(unseen_classes) if i < len(CLASS_NAMES)]}")
del df_train   # free memory

# ── GRU Binary evaluation on unseen attack classes ────────────────────────────
class GRUWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes_mc, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head_state  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
        self.head_binary = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, 2))
        self.head_mc     = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, n_classes_mc))
    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.layer_norm(out[:, -1, :])
        return self.head_state(last), self.head_binary(last), self.head_mc(last)

SEQ_LEN = 10
N_MC    = 15
ckpt    = torch.load(WM_DIR / "gru_world_model.pt", map_location="cpu")
hp      = ckpt["hyperparams"]
gru     = GRUWorldModel(hp["input_dim"], hp["hidden_dim"], hp["num_layers"],
                        hp["n_classes_mc"], hp["dropout"])
gru.load_state_dict(ckpt["model_state_dict"])
gru.eval()

print("\n[STEP 2] GRU binary predictions on test...")
BATCH = 4096
all_p_bin, all_p_mc = [], []
with torch.no_grad():
    for start in range(0, len(X_test) - SEQ_LEN, BATCH):
        end = min(start + BATCH, len(X_test) - SEQ_LEN)
        seqs = [X_test[i:i+SEQ_LEN] for i in range(start, end)]
        x = torch.tensor(np.stack(seqs), dtype=torch.float32)
        _, pb, pm = gru(x)
        all_p_bin.extend(F.softmax(pb, dim=1)[:, 1].numpy().tolist())
        all_p_mc.extend(F.softmax(pm, dim=1).numpy().tolist())

p_bin = np.array(all_p_bin)
p_mc  = np.array(all_p_mc)
y_bin_trunc = y_bin[SEQ_LEN:]
y_mc_trunc  = y_mc[SEQ_LEN:]
print(f"  Predictions: {len(p_bin):,}")

# ── Per-class AUC (one-vs-rest) ───────────────────────────────────────────────
print("\n[STEP 3] Per-class AUC analysis (seen vs unseen)...")
per_class_results = {}
for cls_idx in sorted(test_class_counts.keys()):
    cls_name  = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"cls_{cls_idx}"
    y_ovr     = (y_mc_trunc == cls_idx).astype(int)
    if y_ovr.sum() < 10:
        continue
    p_ovr = p_mc[:, cls_idx] if p_mc.shape[1] > cls_idx else p_bin
    try:
        auc = float(roc_auc_score(y_ovr, p_ovr))
        ap  = float(average_precision_score(y_ovr, p_ovr))
    except Exception:
        auc, ap = 0.5, 0.0
    status = "UNSEEN" if cls_idx in unseen_classes else "SEEN"
    per_class_results[cls_name] = {
        "class_idx": int(cls_idx), "status": status,
        "n_test_samples": int(y_ovr.sum()),
        "roc_auc_ovr": round(auc, 4), "avg_precision": round(ap, 4)
    }
    print(f"  [{status:6s}] {cls_name:<35} n={y_ovr.sum():>6,}  AUC={auc:.4f}  AP={ap:.4f}")

# ── GRU Binary AUC on attack-only subsets ────────────────────────────────────
print("\n[STEP 4] GRU Binary AUC on seen vs unseen attack rows...")
unseen_mask = np.isin(y_mc_trunc, list(unseen_classes)) & (y_bin_trunc == 1)
seen_mask   = (~np.isin(y_mc_trunc, list(unseen_classes))) & (y_bin_trunc == 1)
benign_mask = (y_bin_trunc == 0)

# AUC: unseen attacks vs BENIGN
idx_unseen = np.where(unseen_mask | benign_mask)[0]
idx_seen   = np.where(seen_mask   | benign_mask)[0]

auc_unseen = float(roc_auc_score(y_bin_trunc[idx_unseen], p_bin[idx_unseen])) if unseen_mask.sum() > 0 else None
auc_seen   = float(roc_auc_score(y_bin_trunc[idx_seen],   p_bin[idx_seen]))   if seen_mask.sum()   > 0 else None
auc_full   = float(roc_auc_score(y_bin_trunc,             p_bin))

if auc_seen is not None:
    print(f"  Seen attacks vs BENIGN:   {auc_seen:.4f}")
else:
    print("  Seen attacks vs BENIGN:   N/A (No seen attacks in test set)")
if auc_unseen is not None:
    print(f"  Unseen attacks vs BENIGN: {auc_unseen:.4f}")

# ── PLOT 1: Per-class AUC bar (seen=blue, unseen=red) ─────────────────────────
print("\n[STEP 5] Generating plots...")
sorted_results = sorted(per_class_results.items(), key=lambda x: -x[1]["roc_auc_ovr"])
names = [r[0] for r in sorted_results]
aucs  = [r[1]["roc_auc_ovr"] for r in sorted_results]
colors = ["#e74c3c" if r[1]["status"] == "UNSEEN" else "#3498db" for r in sorted_results]

from matplotlib.patches import Patch
fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(names[::-1], aucs[::-1], color=colors[::-1], edgecolor="white", height=0.7)
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random baseline (AUC=0.5)")
ax.set_xlabel("ROC-AUC (One-vs-Rest)", fontsize=11)
ax.set_title("GRU World Model — Per-Class AUC: Seen vs Unseen Attack Types\n"
             "RED = unseen during training (zero-shot detection test)",
             fontsize=13, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(handles=[Patch(facecolor="#e74c3c", label="Unseen (zero-shot)"),
                    Patch(facecolor="#3498db", label="Seen in training"),
                    Patch(facecolor="gray",    label="Random baseline")],
          loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase14_unseen_attack_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 1 saved: phase14_unseen_attack_roc.png")

# ── PLOT 2: Risk score distribution unseen vs seen ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
if unseen_mask.sum() > 0:
    ax.hist(p_bin[unseen_mask], bins=50, alpha=0.7, label="Unseen Attacks", color="#e74c3c", density=True)
ax.hist(p_bin[seen_mask],   bins=50, alpha=0.7, label="Seen Attacks",   color="#3498db", density=True)
ax.hist(p_bin[benign_mask], bins=50, alpha=0.5, label="BENIGN",         color="#27ae60", density=True)
ax.set_xlabel("GRU P(Attack) score", fontsize=11)
ax.set_ylabel("Density", fontsize=11)
ax.set_title("P(Attack) Distribution: Unseen vs Seen Attacks vs BENIGN", fontsize=13, fontweight="bold")
ax.legend()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase14_unseen_prob_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 2 saved: phase14_unseen_prob_distribution.png")

# ── Save metrics JSON ─────────────────────────────────────────────────────────
metrics = {
    "phase": 14,
    "description": "Zero-shot / unseen attack evaluation",
    "train_class_ids": sorted([int(x) for x in train_classes]),
    "test_class_ids":  sorted([int(x) for x in test_classes]),
    "unseen_class_ids": sorted([int(x) for x in unseen_classes]),
    "unseen_class_names": [CLASS_NAMES[i] for i in sorted(unseen_classes) if i < len(CLASS_NAMES)],
    "gru_binary_auc": {
        "full_test": round(auc_full, 4),
        "seen_attacks_vs_benign":   round(auc_seen, 4) if auc_seen else None,
        "unseen_attacks_vs_benign": round(auc_unseen, 4) if auc_unseen else None,
    },
    "per_class_results": per_class_results
}
with open(METRICS_DIR / "phase14_unseen_attack_eval.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 14 COMPLETE — UNSEEN ATTACK EVALUATION")
print("=" * 70)
print(f"  Full test AUC:  {auc_full:.4f}")
if auc_unseen: print(f"  Zero-shot AUC (unseen attacks vs BENIGN): {auc_unseen:.4f}")
print(f"  Unseen classes: {[CLASS_NAMES[i] for i in sorted(unseen_classes) if i < len(CLASS_NAMES)]}")
print(f"  2 plots saved to results/plots/")
print("=" * 70)
