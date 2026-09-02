"""
PHASE 13 – Explainability: SHAP + Feature Importance
======================================================
Deliverables:
  1. SHAP TreeExplainer on XGBoost Binary  → summary/beeswarm/bar plots
  2. SHAP TreeExplainer on XGBoost MC      → top-class summary plot
  3. SHAP DeepExplainer on GRU World Model → approximate SHAP values
  4. Feature importance comparison: SHAP vs XGB-native-gain vs Grad-Saliency
  5. results/metrics/phase13_shap_metrics.json
  6. 5 publication-quality plots

Rule: NO fabricated metrics. Every number comes from a real SHAP computation.
"""

import json, pickle, warnings, gc
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import shap
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── SHAP / XGBoost version compatibility monkey-patch ────────────────────────
# XGBoost>=2.0 stores base_score as '[4.999E-1]' which shap.TreeExplainer
# can't parse. We monkey-patch SHAP's XGBTreeModelLoader so float() strips
# the brackets before conversion.
def _apply_shap_xgb_patch():
    import builtins
    import shap.explainers._tree as _st

    _orig_init = _st.XGBTreeModelLoader.__init__
    _real_float = builtins.float

    def _safe_float(x):
        if isinstance(x, str):
            x = x.strip()
            if x.startswith('[') and x.endswith(']'):
                inner = x[1:-1].strip()
                # Multiclass: bracketed comma-separated list → take first value
                first = inner.split(',')[0].strip()
                try:    return _real_float(first)
                except: return 0.5   # safe fallback for SHAP offset
        return _real_float(x)

    def _patched_init(self, model):
        builtins.float = _safe_float
        try:
            _orig_init(self, model)
        finally:
            builtins.float = _real_float  # always restore

    _st.XGBTreeModelLoader.__init__ = _patched_init
    print("  [PATCH] SHAP XGBTreeModelLoader patched for XGBoost base_score compatibility.")

_apply_shap_xgb_patch()

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
WM_DIR      = BASE_DIR / "models" / "world_model"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"

SEQ_LEN      = 10
N_CLASSES_MC = 15
N_SHAP_BG    = 200   # background samples for SHAP
N_SHAP_EVAL  = 500   # evaluation samples for SHAP

print("=" * 70)
print("PHASE 13 – EXPLAINABILITY: SHAP + FEATURE IMPORTANCE")
print("=" * 70)

# ── Load feature/label definitions ───────────────────────────────────────────
print("\n[STEP 1] Loading definitions and data...")
with open(PROC_DIR / "feature_list.json")   as f: feat_def   = json.load(f)
with open(PROC_DIR / "label_encoding.json") as f: label_enc  = json.load(f)
with open(MODELS_DIR / "standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)["scaler"]

FEATURE_COLS  = feat_def["all_numeric_features"]
ENG_FEATS     = set(feat_def["engineered_features"])
CLASS_NAMES   = label_enc["classes"]
D             = len(FEATURE_COLS)

# Load test data (flat — for XGB SHAP)
df_test = pd.read_parquet(PROC_DIR / "state_test.parquet")
df_test = df_test.sort_values(["day_order", "seq_idx"]).reset_index(drop=True)

# Scale features
X_test_raw = df_test[FEATURE_COLS].values.astype(np.float32)
y_bin      = df_test["next_label_binary"].values.astype(int)
y_mc       = df_test["next_label_multiclass"].values.astype(int)
X_test     = scaler.transform(X_test_raw).astype(np.float32)

# Subsample for SHAP (representative: equal BENIGN/ATTACK)
atk_idx = np.where(y_bin == 1)[0]
ben_idx = np.where(y_bin == 0)[0]
rng = np.random.default_rng(42)

eval_idx = np.concatenate([
    rng.choice(atk_idx, size=min(N_SHAP_EVAL//2, len(atk_idx)), replace=False),
    rng.choice(ben_idx, size=min(N_SHAP_EVAL//2, len(ben_idx)), replace=False)
])
bg_idx = np.concatenate([
    rng.choice(atk_idx, size=min(N_SHAP_BG//2, len(atk_idx)), replace=False),
    rng.choice(ben_idx, size=min(N_SHAP_BG//2, len(ben_idx)), replace=False)
])

X_eval = X_test[eval_idx]
X_bg   = X_test[bg_idx]
y_eval = y_bin[eval_idx]
print(f"  Test rows: {len(df_test):,}  | SHAP eval: {len(X_eval)}  | SHAP bg: {len(X_bg)}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: SHAP TreeExplainer — XGBoost Binary
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 2] SHAP TreeExplainer – XGBoost Binary...")
# xgb_binary.pkl is a pickled sklearn XGBClassifier
with open(MODELS_DIR / "xgb_binary.pkl", "rb") as f:
    xgb_bin_model = pickle.load(f)
print(f"  XGB binary model type: {type(xgb_bin_model).__name__}")

# Fix XGBoost+SHAP base_score compatibility: save JSON, patch, reload
import tempfile, os, re
_bst_tmp = xgb_bin_model.get_booster()
with tempfile.NamedTemporaryFile(suffix='.json', delete=False, mode='wb') as _tf:
    _tmp_path = _tf.name
_bst_tmp.save_model(_tmp_path)
with open(_tmp_path, 'r') as _f: _model_txt = _f.read()
def _fix_bs(m):
    try:    return f'"base_score":"{float(m.group(1))}"'
    except: return m.group(0)
_model_txt = re.sub(r'"base_score":\"\[([^\]]+)\]\"', _fix_bs, _model_txt)
with open(_tmp_path, 'w') as _f: _f.write(_model_txt)
_bst_bin = xgb.Booster(); _bst_bin.load_model(_tmp_path); os.unlink(_tmp_path)
_bst_bin.feature_names = FEATURE_COLS   # restore feature names

explainer_bin = shap.TreeExplainer(_bst_bin, feature_perturbation="tree_path_dependent")
shap_vals_bin = explainer_bin.shap_values(X_eval)

# TreeExplainer on booster returns (N, D) for binary log-odds
shap_attack = np.array(shap_vals_bin)
if shap_attack.ndim == 3:          # (2, N, D) — take class 1
    shap_attack = shap_attack[1]
elif shap_attack.ndim == 2 and shap_attack.shape[0] == 2:   # (2, N*D)
    shap_attack = shap_attack[1]
print(f"  SHAP binary computed: shape={shap_attack.shape}")

# Mean absolute SHAP (global importance)
mean_abs_shap_bin = np.abs(shap_attack).mean(axis=0)
top20_bin_idx     = np.argsort(mean_abs_shap_bin)[-20:][::-1]
top20_bin_feats   = [FEATURE_COLS[i] for i in top20_bin_idx]
top20_bin_vals    = [float(mean_abs_shap_bin[i]) for i in top20_bin_idx]

print(f"  Top SHAP feature (binary): {top20_bin_feats[0]}  = {top20_bin_vals[0]:.4f}")

# ── PLOT 1: SHAP Summary (dot/beeswarm) XGB Binary ───────────────────────────
print("  Generating SHAP beeswarm plot...")
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_attack, X_eval, feature_names=FEATURE_COLS,
                  max_display=15, show=False, plot_type="dot",
                  color_bar=True)
plt.title("SHAP Summary — XGBoost Binary (ATTACK class)\n"
          "Each dot = one sample. Color = feature value. X-axis = SHAP impact.",
          fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase13_shap_xgb_binary_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 1 saved: phase13_shap_xgb_binary_beeswarm.png")

# ── PLOT 2: SHAP Bar (mean |SHAP|) — styled ───────────────────────────────────
print("  Generating SHAP bar plot...")
colors_bar = ["#e74c3c" if FEATURE_COLS[i] in ENG_FEATS else "#2980b9" for i in top20_bin_idx]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_bin_feats[::-1], top20_bin_vals[::-1],
        color=colors_bar[::-1], edgecolor="white", height=0.7)
ax.set_xlabel("Mean |SHAP Value| — Impact on ATTACK Probability", fontsize=11)
ax.set_title("SHAP Feature Importance — XGBoost Binary\nTop 20 Features by Mean Absolute SHAP",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(handles=[Patch(facecolor="#e74c3c", label="Engineered Feature"),
                    Patch(facecolor="#2980b9", label="Original Feature")],
          loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase13_shap_xgb_binary_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 2 saved: phase13_shap_xgb_binary_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: SHAP TreeExplainer — XGBoost Multiclass
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 3] SHAP TreeExplainer – XGBoost Multiclass...")
bst_mc = xgb.Booster()
bst_mc.load_model(str(MODELS_DIR / "xgb_multiclass.json"))
bst_mc.feature_names = FEATURE_COLS   # ensure feature names are set
# (monkey-patch already applied globally — no extra fix needed)
explainer_mc  = shap.TreeExplainer(bst_mc, feature_perturbation="interventional")

shap_vals_mc  = explainer_mc.shap_values(X_eval)

# shap_vals_mc for multiclass booster — list of (N, D) per class, or (N, D, C)
arr_mc = np.array(shap_vals_mc)
print(f"  raw shap_vals_mc shape: {arr_mc.shape}")
if arr_mc.ndim == 3 and arr_mc.shape[0] == N_CLASSES_MC:  # (C, N, D)
    mean_abs_shap_mc = np.abs(arr_mc).mean(axis=(0, 1))
elif arr_mc.ndim == 3 and arr_mc.shape[2] == N_CLASSES_MC:  # (N, D, C)
    mean_abs_shap_mc = np.abs(arr_mc).mean(axis=(0, 2))
elif isinstance(shap_vals_mc, list):
    mean_abs_shap_mc = np.mean([np.abs(v).mean(axis=0) for v in shap_vals_mc], axis=0)
else:
    mean_abs_shap_mc = np.abs(arr_mc).mean(axis=0)

# Ensure 1D shape of length D and int indices
mean_abs_shap_mc = np.array(mean_abs_shap_mc).flatten()[:D]
top20_mc_idx   = np.argsort(mean_abs_shap_mc)[-20:][::-1].astype(int).tolist()
top20_mc_feats = [FEATURE_COLS[i] for i in top20_mc_idx]
top20_mc_vals  = [float(mean_abs_shap_mc[i]) for i in top20_mc_idx]
print(f"  SHAP MC computed: mean_abs shape={mean_abs_shap_mc.shape}")
print(f"  Top MC SHAP feature: {top20_mc_feats[0]}  = {top20_mc_vals[0]:.5f}")


# ── PLOT 3: SHAP MC bar ────────────────────────────────────────────────────────
print("  Generating SHAP MC bar plot...")
colors_mc = ["#e74c3c" if FEATURE_COLS[i] in ENG_FEATS else "#2980b9" for i in top20_mc_idx]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_mc_feats[::-1], top20_mc_vals[::-1],
        color=colors_mc[::-1], edgecolor="white", height=0.7)
ax.set_xlabel("Mean |SHAP Value| across all attack classes", fontsize=11)
ax.set_title("SHAP Feature Importance — XGBoost Multiclass\nTop 20 Features (Averaged Across All Classes)",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(handles=[Patch(facecolor="#e74c3c", label="Engineered Feature"),
                    Patch(facecolor="#2980b9", label="Original Feature")],
          loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase13_shap_xgb_mc_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 3 saved: phase13_shap_xgb_mc_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: SHAP DeepExplainer — GRU World Model
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 4] SHAP DeepExplainer – GRU World Model (binary head)...")

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
        return self.head_binary(last)   # only binary for SHAP (returns logits)

ckpt  = torch.load(WM_DIR / "gru_world_model.pt", map_location="cpu")
hp    = ckpt["hyperparams"]
gru   = GRUWorldModel(hp["input_dim"], hp["hidden_dim"], hp["num_layers"],
                       hp["n_classes_mc"], hp["dropout"])
gru.load_state_dict(ckpt["model_state_dict"], strict=False)
gru.eval()

# Build sequences for SHAP: shape (N, SEQ_LEN, D)
# Use consecutive rows from test set that don't cross day/boundary boundaries
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, scaler, stride=1):
        fa = scaler.transform(df[feature_cols].values.astype(np.float32)).astype(np.float32)
        self.yb   = df["next_label_binary"].values.astype(np.int64)
        day       = df["day_order"].values
        bnd       = df["is_boundary"].values
        n         = len(df)
        valid     = []
        i         = 0
        while i + seq_len < n:
            e = i + seq_len
            if day[i] != day[e] or np.any(bnd[i:e]):
                i += 1; continue
            valid.append(i); i += stride
        self.idx = np.array(valid, dtype=np.int64)
        self.fa  = fa
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s = self.idx[i]; e = s + SEQ_LEN
        return torch.from_numpy(self.fa[s:e].copy()), int(self.yb[e-1])

seq_ds = SeqDataset(df_test, SEQ_LEN, FEATURE_COLS, scaler, stride=10)
seq_yb = np.array([seq_ds[i][1] for i in range(len(seq_ds))])

# Pick balanced background and eval sequences
seq_atk = np.where(seq_yb == 1)[0]
seq_ben = np.where(seq_yb == 0)[0]
N_BG_SEQ   = 50
N_EVAL_SEQ = 100

bg_seq_idx   = np.concatenate([
    rng.choice(seq_atk, size=min(N_BG_SEQ//2, len(seq_atk)), replace=False),
    rng.choice(seq_ben, size=min(N_BG_SEQ//2, len(seq_ben)), replace=False)
])
eval_seq_idx = np.concatenate([
    rng.choice(seq_atk, size=min(N_EVAL_SEQ//2, len(seq_atk)), replace=False),
    rng.choice(seq_ben, size=min(N_EVAL_SEQ//2, len(seq_ben)), replace=False)
])

X_bg_seq   = torch.stack([seq_ds[i][0] for i in bg_seq_idx])    # (50,  10, D)
X_eval_seq = torch.stack([seq_ds[i][0] for i in eval_seq_idx])  # (100, 10, D)
y_eval_seq = seq_yb[eval_seq_idx]

print(f"  GRU SHAP  bg: {X_bg_seq.shape}  eval: {X_eval_seq.shape}")

# DeepExplainer on GRU (treats last-step features as the input space)
explainer_gru = shap.DeepExplainer(gru, X_bg_seq)
shap_vals_gru = explainer_gru.shap_values(X_eval_seq)

# shap_vals_gru: list of 2 (for 2 classes) each (N, SEQ_LEN, D)
# Take class-1 (ATTACK), average over time steps
if isinstance(shap_vals_gru, list):
    shap_gru_atk = np.array(shap_vals_gru[1])   # (N, 10, D)
else:
    shap_gru_atk = np.array(shap_vals_gru)

shap_gru_mean = np.abs(shap_gru_atk).mean(axis=(0, 1))   # (D,)
top20_gru_idx   = np.argsort(shap_gru_mean)[-20:][::-1]
top20_gru_feats = [FEATURE_COLS[i] for i in top20_gru_idx]
top20_gru_vals  = [float(shap_gru_mean[i]) for i in top20_gru_idx]
print(f"  GRU SHAP computed. Top feature: {top20_gru_feats[0]}  = {top20_gru_vals[0]:.5f}")

# ── PLOT 4: GRU SHAP bar ──────────────────────────────────────────────────────
print("  Generating GRU SHAP bar plot...")
colors_gru = ["#e74c3c" if FEATURE_COLS[i] in ENG_FEATS else "#9b59b6" for i in top20_gru_idx]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(top20_gru_feats[::-1], top20_gru_vals[::-1],
        color=colors_gru[::-1], edgecolor="white", height=0.7)
ax.set_xlabel("Mean |SHAP Value| (DeepExplainer, averaged over 10 time steps)", fontsize=10)
ax.set_title("SHAP Feature Importance — GRU World Model (Binary Head)\n"
             "DeepExplainer | Top 20 Features",
             fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.legend(handles=[Patch(facecolor="#e74c3c", label="Engineered"),
                    Patch(facecolor="#9b59b6", label="Original")],
          loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase13_shap_gru_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 4 saved: phase13_shap_gru_bar.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: Three-way comparison — XGB SHAP vs GRU SHAP vs Gradient Saliency
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 5] Three-way feature importance comparison plot...")

# Gradient saliency from Phase 10 (already in metrics JSON)
with open(METRICS_DIR / "phase10_forecasting_metrics.json") as f:
    ph10 = json.load(f)
sal_feats = {x["feature"]: x["saliency"] for x in ph10["top20_attack_saliency_features"]}

# XGB native gain from Phase 7
with open(METRICS_DIR / "phase7_xgb_metrics.json") as f:
    ph7 = json.load(f)
gain_feats = {x["feature"]: x["gain"] for x in ph7["multiclass"]["top15_features_gain"]}
max_gain   = max(gain_feats.values()) if gain_feats else 1.0

# Top 15 features by XGB SHAP binary
top15_feats = top20_bin_feats[:15]

# Normalize each method 0→1 for comparison
shap_xgb_v  = np.array([mean_abs_shap_bin[FEATURE_COLS.index(f)] if f in FEATURE_COLS else 0 for f in top15_feats])
shap_gru_v  = np.array([shap_gru_mean[FEATURE_COLS.index(f)]     if f in FEATURE_COLS else 0 for f in top15_feats])
sal_v       = np.array([sal_feats.get(f, 0) for f in top15_feats])
gain_v      = np.array([gain_feats.get(f, 0) / max_gain for f in top15_feats])

# Normalize 0-1
def norm(x): return (x - x.min()) / (x.max() - x.min() + 1e-12)
shap_xgb_n = norm(shap_xgb_v)
shap_gru_n = norm(shap_gru_v)
sal_n      = norm(sal_v)
gain_n     = norm(gain_v)

y_pos   = np.arange(len(top15_feats))
bar_h   = 0.18
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(y_pos + bar_h*1.5, shap_xgb_n, height=bar_h, label="SHAP (XGBoost)",       color="#e74c3c", alpha=0.9)
ax.barh(y_pos + bar_h*0.5, shap_gru_n, height=bar_h, label="SHAP (GRU DeepExpl.)", color="#9b59b6", alpha=0.9)
ax.barh(y_pos - bar_h*0.5, sal_n,      height=bar_h, label="Gradient Saliency (GRU)", color="#3498db", alpha=0.9)
ax.barh(y_pos - bar_h*1.5, gain_n,     height=bar_h, label="XGBoost Gain (native)",  color="#27ae60", alpha=0.9)
ax.set_yticks(y_pos); ax.set_yticklabels(top15_feats, fontsize=9)
ax.set_xlabel("Normalized Importance [0–1]", fontsize=11)
ax.set_title("Feature Importance Comparison\nSHAP (XGBoost) | SHAP (GRU) | Grad Saliency | XGBoost Gain",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase13_feature_importance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Plot 5 saved: phase13_feature_importance_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: Save metrics JSON
# ══════════════════════════════════════════════════════════════════════════════
print("\n[STEP 6] Saving phase13_shap_metrics.json...")

shap_metrics = {
    "phase": 13,
    "description": "SHAP Explainability — XGBoost (TreeExplainer) + GRU (DeepExplainer)",
    "n_shap_eval_samples": int(len(X_eval)),
    "n_shap_bg_samples":   int(len(X_bg)),
    "n_gru_shap_eval_seq": int(len(X_eval_seq)),
    "n_gru_shap_bg_seq":   int(len(X_bg_seq)),

    "xgb_binary_shap": {
        "explainer": "TreeExplainer",
        "top20_features": [
            {"rank": i+1, "feature": top20_bin_feats[i],
             "mean_abs_shap": round(top20_bin_vals[i], 6),
             "engineered": top20_bin_feats[i] in ENG_FEATS}
            for i in range(20)
        ],
        "engineered_in_top10": int(sum(1 for f in top20_bin_feats[:10] if f in ENG_FEATS)),
    },

    "xgb_mc_shap": {
        "explainer": "TreeExplainer",
        "top20_features": [
            {"rank": i+1, "feature": top20_mc_feats[i],
             "mean_abs_shap": round(top20_mc_vals[i], 6),
             "engineered": top20_mc_feats[i] in ENG_FEATS}
            for i in range(20)
        ],
        "engineered_in_top10": int(sum(1 for f in top20_mc_feats[:10] if f in ENG_FEATS)),
    },

    "gru_shap": {
        "explainer": "DeepExplainer",
        "seq_len_used": SEQ_LEN,
        "top20_features": [
            {"rank": i+1, "feature": top20_gru_feats[i],
             "mean_abs_shap_over_time": round(top20_gru_vals[i], 6),
             "engineered": top20_gru_feats[i] in ENG_FEATS}
            for i in range(20)
        ],
        "engineered_in_top10": int(sum(1 for f in top20_gru_feats[:10] if f in ENG_FEATS)),
    },

    "key_findings": {
        "xgb_top1_feature": top20_bin_feats[0],
        "gru_top1_feature":  top20_gru_feats[0],
        "engineered_in_top5_xgb": int(sum(1 for f in top20_bin_feats[:5] if f in ENG_FEATS)),
        "engineered_in_top5_gru": int(sum(1 for f in top20_gru_feats[:5] if f in ENG_FEATS)),
        "agreement_top10": int(len(set(top20_bin_feats[:10]) & set(top20_gru_feats[:10]))),
    },

    "plots_generated": [
        "phase13_shap_xgb_binary_beeswarm.png",
        "phase13_shap_xgb_binary_bar.png",
        "phase13_shap_xgb_mc_bar.png",
        "phase13_shap_gru_bar.png",
        "phase13_feature_importance_comparison.png",
    ]
}

with open(METRICS_DIR / "phase13_shap_metrics.json", "w") as f:
    json.dump(shap_metrics, f, indent=2)

print("  phase13_shap_metrics.json saved.")

# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHASE 13 COMPLETE — SHAP EXPLAINABILITY")
print("=" * 70)
print(f"\n  XGBoost Binary (TreeExplainer):")
print(f"    Top 1: {top20_bin_feats[0]:<40} SHAP={top20_bin_vals[0]:.5f}")
print(f"    Top 2: {top20_bin_feats[1]:<40} SHAP={top20_bin_vals[1]:.5f}")
print(f"    Engineered in top-10: {shap_metrics['xgb_binary_shap']['engineered_in_top10']}/10")

print(f"\n  GRU World Model (DeepExplainer):")
print(f"    Top 1: {top20_gru_feats[0]:<40} SHAP={top20_gru_vals[0]:.5f}")
print(f"    Top 2: {top20_gru_feats[1]:<40} SHAP={top20_gru_vals[1]:.5f}")
print(f"    Engineered in top-10: {shap_metrics['gru_shap']['engineered_in_top10']}/10")

print(f"\n  Feature agreement (top-10 overlap, XGB∩GRU): {shap_metrics['key_findings']['agreement_top10']} features")
print(f"\n  5 plots saved to results/plots/")
print("=" * 70)
