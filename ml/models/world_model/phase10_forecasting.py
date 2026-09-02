"""
PHASE 10 – K-step Rollout, MITRE ATT&CK Mapping, Explainability & Risk Scoring
================================================================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

COMPONENTS:
  1. K-step Autoregressive Rollout
     • Load GRU World Model (best binary model from Phase 8)
     • Given current sequence [S_{t-L+1}..S_t], autoregressively predict
       Ŝ_{t+1}, Ŝ_{t+2}, ..., Ŝ_{t+K} by feeding predicted states back
     • Track P(attack at t+k) for k=1..K (attack probability timeline)

  2. Rollout Quality Evaluation (vs real labels)
     • AUC at each rollout step k — measures forecasting accuracy decay
     • Compare "k=1 rollout" vs "direct prediction" to verify rollout fidelity

  3. MITRE ATT&CK Mapping
     • Map predicted multiclass probabilities → MITRE tactic distribution
     • Track tactic probability across rollout horizon
     • Identify most-likely next MITRE stage

  4. Risk Score
     • risk_score = Σ_k [ discount^k × P(attack at t+k) × severity_weight ]
     • Normalized to [0, 100]
     • Threshold analysis: low/medium/high/critical bands

  5. Explainability (Gradient Saliency)
     • ∂P(ATTACK)/∂x_f averaged over all sequence timesteps
     • Top-K features driving the attack prediction
     • Comparison: BENIGN vs ATTACK sequence saliency profiles

  6. Attack Progression Forecasting Demo
     • Select benign→attack transition sequences from test set
     • Show P(attack) timeline: model "sees" attack before it happens
     • MITRE stage progression visualization

STRICT RULES:
  - Uses GRU best checkpoint (Phase 8) — no re-training
  - All metrics from real test labels — ZERO fabrication
  - Rollout uses ONLY scaled predicted states (no inverse-transform cheating)
  - Fixed seed = 42
"""

import sys, json, warnings, time, pickle, gc
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")
SEED = 42; torch.manual_seed(SEED); np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR/"data"/"processed"
MODELS_DIR  = BASE_DIR/"models"
WM_DIR      = BASE_DIR/"models"/"world_model"
METRICS_DIR = BASE_DIR/"results"/"metrics"
PLOTS_DIR   = BASE_DIR/"results"/"plots"
INFER_DIR   = BASE_DIR/"ml"/"inference"
sys.path.insert(0, str(BASE_DIR/"ml"/"inference"))

# ── Hyperparameters ───────────────────────────────────────────────────────────
SEQ_LEN      = 10
ROLLOUT_K    = 8        # steps ahead to forecast
N_EVAL       = 3000     # test sequences for rollout evaluation
RISK_DISCOUNT= 0.85     # discount factor for future risk
RISK_THRESH  = {"low":20, "medium":40, "high":65, "critical":85}
N_CLASSES_MC = 15
D            = 60


# ══════════════════════════════════════════════════════════════════════════════
# GRU MODEL (reconstructed from Phase 8)
# ══════════════════════════════════════════════════════════════════════════════
class GRUWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes_mc, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head_state  = nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),
                                          nn.Dropout(dropout),nn.Linear(hidden_dim,input_dim))
        self.head_binary = nn.Sequential(nn.Linear(hidden_dim,64),nn.ReLU(),
                                          nn.Dropout(dropout),nn.Linear(64,2))
        self.head_mc     = nn.Sequential(nn.Linear(hidden_dim,64),nn.ReLU(),
                                          nn.Dropout(dropout),nn.Linear(64,n_classes_mc))

    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.layer_norm(out[:,-1,:])
        return self.head_state(last), self.head_binary(last), self.head_mc(last)


# ══════════════════════════════════════════════════════════════════════════════
# ROLLOUT ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class RolloutEngine:
    """
    K-step autoregressive rollout using the GRU world model.

    At each step:
      1. Feed current window [S_{t-L+1}..S_t] through GRU
      2. Get: Ŝ_{t+1} (next state), P_bin (attack prob), P_mc (class probs)
      3. Slide window: drop oldest, append Ŝ_{t+1}
      4. Repeat K times
    """
    def __init__(self, model, device, seq_len=10):
        self.model   = model
        self.device  = device
        self.seq_len = seq_len
        self.model.eval()

    @torch.no_grad()
    def rollout(self, initial_seq: np.ndarray, K: int):
        """
        Args:
            initial_seq: shape (seq_len, D) — scaled initial window
            K: number of rollout steps
        Returns:
            attack_probs: (K,)      — P(ATTACK) at each step
            class_probs:  (K, n_mc) — P(class) at each step
            next_states:  (K, D)    — predicted next states (scaled)
        """
        window = initial_seq.copy()   # (seq_len, D)
        attack_probs = []
        class_probs  = []
        next_states  = []

        for k in range(K):
            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            pred_state, pred_bin, pred_mc = self.model(x)

            p_bin = F.softmax(pred_bin, dim=1).squeeze(0).cpu().numpy()  # (2,)
            p_mc  = F.softmax(pred_mc,  dim=1).squeeze(0).cpu().numpy()  # (15,)
            s_hat = pred_state.squeeze(0).cpu().numpy()                   # (D,)

            attack_probs.append(float(p_bin[1]))
            class_probs.append(p_mc)
            next_states.append(s_hat)

            # Slide window: drop oldest, append predicted next state
            window = np.vstack([window[1:], s_hat[np.newaxis, :]])

        return (np.array(attack_probs),
                np.array(class_probs),
                np.array(next_states))

    @torch.no_grad()
    def batch_single_step(self, X_batch: torch.Tensor):
        """Direct single-step prediction (no rollout) — for baseline comparison."""
        X_batch = X_batch.to(self.device)
        _, pred_bin, pred_mc = self.model(X_batch)
        p_bin = F.softmax(pred_bin, dim=1).cpu().numpy()
        p_mc  = F.softmax(pred_mc,  dim=1).cpu().numpy()
        return p_bin, p_mc


# ══════════════════════════════════════════════════════════════════════════════
# RISK SCORER
# ══════════════════════════════════════════════════════════════════════════════
class RiskScorer:
    """
    Computes a discounted, severity-weighted risk score from rollout probabilities.

    risk_score = Σ_k [ γ^k × P(attack at t+k) × severity_k ] × 100
                 normalized to [0, 100]

    severity_k = max severity weight of top predicted class at step k
    γ = discount factor (future steps weighted less)
    """
    SEVERITY_BY_CLASS = {
        0:  0,   # BENIGN
        1:  5,   # Bot
        2:  5,   # DDoS
        3:  4,   # DoS_GoldenEye
        4:  4,   # DoS_Hulk
        5:  3,   # DoS_Slowhttptest
        6:  3,   # DoS_slowloris
        7:  3,   # FTP-Patator
        8:  5,   # Heartbleed
        9:  5,   # Infiltration
        10: 2,   # PortScan
        11: 3,   # SSH-Patator
        12: 3,   # WebAttack_BruteForce
        13: 5,   # WebAttack_SQL_Injection
        14: 3,   # WebAttack_XSS
    }

    def __init__(self, discount=0.85):
        self.discount = discount

    def score(self, attack_probs: np.ndarray, class_probs: np.ndarray) -> float:
        K    = len(attack_probs)
        dmax = sum(self.discount**k for k in range(K)) * 5.0   # theoretical max (sev=5, p=1)
        raw  = 0.0
        for k in range(K):
            top_class  = int(np.argmax(class_probs[k]))
            severity   = self.SEVERITY_BY_CLASS.get(top_class, 0) / 5.0
            weighted   = attack_probs[k] * (1 + severity)  # range [0, 2]
            raw       += (self.discount ** k) * weighted
        score = min(raw / (dmax * 2.0 / 5.0) * 100, 100)   # normalized to [0,100]
        return round(float(score), 2)

    def classify_risk(self, score: float) -> str:
        if score >= RISK_THRESH["critical"]: return "CRITICAL"
        if score >= RISK_THRESH["high"]:     return "HIGH"
        if score >= RISK_THRESH["medium"]:   return "MEDIUM"
        if score >= RISK_THRESH["low"]:      return "LOW"
        return "SAFE"


# ══════════════════════════════════════════════════════════════════════════════
# GRADIENT SALIENCY EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
class GradientSaliency:
    """
    Computes input × gradient saliency for P(ATTACK).
    Identifies which features most strongly drive the attack prediction.
    """
    def __init__(self, model, device):
        self.model  = model
        self.device = device

    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: (batch, seq_len, D)
        Returns:
            saliency: (D,) — mean |grad| × |input| across batch and timesteps
        """
        x_t = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(self.device)
        self.model.zero_grad()
        _, pred_bin, _ = self.model(x_t)
        attack_score   = F.softmax(pred_bin, dim=1)[:, 1].sum()
        attack_score.backward()
        # Gradient × input (integrated gradients approximation)
        saliency = (x_t.grad.abs() * x_t.abs()).detach().cpu().numpy()
        # Average over batch and sequence timesteps → shape (D,)
        return saliency.mean(axis=(0, 1))


# ══════════════════════════════════════════════════════════════════════════════
# DATASET (same as Phase 8/9)
# ══════════════════════════════════════════════════════════════════════════════
class NetworkSequenceDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols, scaler=None, stride=1):
        self.seq_len = seq_len
        next_cols    = [f"next_{c}" for c in feature_cols]
        feat_arr      = df[feature_cols].values.astype(np.float32)
        next_feat_arr = df[next_cols].values.astype(np.float32)
        if scaler:
            feat_arr      = scaler.transform(feat_arr).astype(np.float32)
            next_feat_arr = scaler.transform(next_feat_arr).astype(np.float32)
        self.feat_arr      = feat_arr
        self.next_feat_arr = next_feat_arr
        self.y_bin_arr     = df["next_label_binary"].values.astype(np.int64)
        self.y_mc_arr      = df["next_label_multiclass"].values.astype(np.int64)
        day_orders = df["day_order"].values
        is_boundary= df["is_boundary"].values
        n = len(df); valid = []
        i = 0
        while i + seq_len < n:
            end = i + seq_len
            if day_orders[i] != day_orders[end]: i+=1; continue
            if np.any(is_boundary[i:end]):        i+=1; continue
            valid.append(i); i += stride
        self.indices = np.array(valid, dtype=np.int64)

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]; e = s + self.seq_len
        return (torch.from_numpy(self.feat_arr[s:e].copy()),
                torch.from_numpy(self.next_feat_arr[e-1].copy()),
                torch.tensor(int(self.y_bin_arr[e-1]), dtype=torch.long),
                torch.tensor(int(self.y_mc_arr[e-1]),  dtype=torch.long))


# ══════════════════════════════════════════════════════════════════════════════
# MITRE MAPPING (inline — from ml/inference/mitre_mapper.py)
# ══════════════════════════════════════════════════════════════════════════════
MITRE_MAPPING = {
    0:  {"label":"BENIGN",               "tactic":None,                    "stage":0,"severity":0},
    1:  {"label":"Bot",                  "tactic":"Command & Control",      "stage":6,"severity":5},
    2:  {"label":"DDoS",                 "tactic":"Impact",                 "stage":7,"severity":5},
    3:  {"label":"DoS_GoldenEye",        "tactic":"Impact",                 "stage":7,"severity":4},
    4:  {"label":"DoS_Hulk",             "tactic":"Impact",                 "stage":7,"severity":4},
    5:  {"label":"DoS_Slowhttptest",     "tactic":"Impact",                 "stage":7,"severity":3},
    6:  {"label":"DoS_slowloris",        "tactic":"Impact",                 "stage":7,"severity":3},
    7:  {"label":"FTP-Patator",          "tactic":"Credential Access",      "stage":3,"severity":3},
    8:  {"label":"Heartbleed",           "tactic":"Initial Access",         "stage":2,"severity":5},
    9:  {"label":"Infiltration",         "tactic":"Lateral Movement",       "stage":5,"severity":5},
    10: {"label":"PortScan",             "tactic":"Discovery",              "stage":1,"severity":2},
    11: {"label":"SSH-Patator",          "tactic":"Credential Access",      "stage":3,"severity":3},
    12: {"label":"WebAttack_BruteForce", "tactic":"Credential Access",      "stage":3,"severity":3},
    13: {"label":"WebAttack_SQL_Inj",    "tactic":"Initial Access",         "stage":2,"severity":5},
    14: {"label":"WebAttack_XSS",        "tactic":"Execution",              "stage":4,"severity":3},
}
TACTIC_COLORS = {
    None:"#95a5a6","Discovery":"#3498db","Initial Access":"#e67e22",
    "Credential Access":"#e74c3c","Execution":"#9b59b6","Lateral Movement":"#1abc9c",
    "Command & Control":"#c0392b","Impact":"#922b21",
}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def plot_rollout_auc(rollout_aucs, single_step_auc, path):
    steps = list(range(1, len(rollout_aucs)+1))
    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(steps, rollout_aucs, "o-", color="#e74c3c", lw=2.5,
            markersize=7, label="Rollout AUC (k-step ahead)")
    ax.axhline(single_step_auc, color="#2980b9", ls="--", lw=2,
               label=f"Single-step direct AUC ({single_step_auc:.3f})")
    ax.axhline(0.5, color="#95a5a6", ls=":", lw=1.5, label="Random baseline (0.5)")
    for i,auc in enumerate(rollout_aucs):
        ax.annotate(f"{auc:.3f}", (steps[i], auc),
                    textcoords="offset points", xytext=(0,8), fontsize=8, ha="center")
    ax.set_xlabel("Rollout Horizon k (steps ahead)", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("World Model — Rollout Forecast AUC vs Horizon", fontsize=14, fontweight="bold")
    ax.set_ylim([0.45, 1.0]); ax.set_xticks(steps)
    ax.legend(fontsize=10); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_risk_distribution(risk_scores, risk_labels, path):
    colors_map = {"SAFE":"#27ae60","LOW":"#f1c40f","MEDIUM":"#e67e22",
                  "HIGH":"#e74c3c","CRITICAL":"#8e44ad"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
    # Histogram
    ax1.hist(risk_scores, bins=50, color="#2980b9", edgecolor="white", alpha=0.8)
    for thresh_name, thresh_val in RISK_THRESH.items():
        ax1.axvline(thresh_val, color=colors_map.get(thresh_name.upper(),"grey"),
                    ls="--", lw=1.5, label=f"{thresh_name.title()} ({thresh_val})")
    ax1.set_xlabel("Risk Score [0-100]", fontsize=11)
    ax1.set_ylabel("Sequence Count", fontsize=11)
    ax1.set_title("Risk Score Distribution", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    # Pie
    from collections import Counter
    counts = Counter(risk_labels)
    order  = ["SAFE","LOW","MEDIUM","HIGH","CRITICAL"]
    vals   = [counts.get(l,0) for l in order]
    cols   = [colors_map[l] for l in order]
    non_zero = [(v,l,c) for v,l,c in zip(vals,order,cols) if v>0]
    if non_zero:
        ax2.pie([x[0] for x in non_zero],
                labels=[x[1] for x in non_zero],
                colors=[x[2] for x in non_zero],
                autopct="%1.1f%%", startangle=140, pctdistance=0.85)
    ax2.set_title("Risk Level Distribution", fontsize=13, fontweight="bold")
    plt.suptitle("GRU World Model — Risk Score Analysis", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_attack_progression_timeline(sequences_info, CLASS_NAMES, path):
    """Plot P(attack) timeline + MITRE tactic for 6 representative sequences."""
    n = min(6, len(sequences_info))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    steps = list(range(1, ROLLOUT_K+1))
    for idx, (ax, sinfo) in enumerate(zip(axes.flatten(), sequences_info[:n])):
        atk_probs   = sinfo["attack_probs"]
        class_prbs  = sinfo["class_probs"]
        true_label  = sinfo["true_label_binary"]
        risk_score  = sinfo["risk_score"]
        risk_level  = sinfo["risk_level"]
        # Color by actual label
        fill_color  = "#e74c3c" if true_label == 1 else "#27ae60"
        ax.plot(steps, atk_probs, "o-", color=fill_color, lw=2.5, markersize=6)
        ax.fill_between(steps, 0, atk_probs, alpha=0.15, color=fill_color)
        ax.axhline(0.5, color="#7f8c8d", ls="--", lw=1, alpha=0.7)
        # Annotate top predicted MITRE tactic at each step
        for k, (prob, cp) in enumerate(zip(atk_probs, class_prbs)):
            top_cls = int(np.argmax(cp))
            tactic  = MITRE_MAPPING[top_cls]["tactic"]
            color   = TACTIC_COLORS.get(tactic, "#95a5a6")
            ax.scatter([steps[k]], [prob], color=color, s=80, zorder=5)
        true_str = "ATTACK ⚠" if true_label==1 else "BENIGN ✓"
        ax.set_title(f"Seq {idx+1}: {true_str} | Risk={risk_score:.1f} ({risk_level})",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Rollout Step k", fontsize=8)
        ax.set_ylabel("P(ATTACK)", fontsize=8)
        ax.set_ylim([0, 1.05]); ax.set_xticks(steps)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Legend for MITRE tactics
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker="o", color="w", markerfacecolor=c,
                       markersize=8, label=t if t else "BENIGN")
               for t,c in TACTIC_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               title="MITRE Tactic (dot color)", bbox_to_anchor=(0.5,-0.02))
    plt.suptitle(f"GRU World Model — {ROLLOUT_K}-Step Attack Progression Forecasting",
                  fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0,0.08,1,1])
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_mitre_tactic_distribution(class_probs_all, path, CLASS_NAMES):
    """Heatmap: mean P(class) at each rollout step across all test sequences."""
    # class_probs_all: (N, K, 15)
    mean_probs = class_probs_all.mean(axis=0)  # (K, 15)
    # Map classes to tactics
    tactic_names = [MITRE_MAPPING[i]["tactic"] or "BENIGN" for i in range(N_CLASSES_MC)]
    steps = [f"k={k+1}" for k in range(ROLLOUT_K)]
    fig, ax = plt.subplots(figsize=(14,6))
    sns.heatmap(mean_probs.T, xticklabels=steps, yticklabels=CLASS_NAMES,
                cmap="YlOrRd", ax=ax, vmin=0, vmax=0.15, linewidths=0.3,
                annot=True, fmt=".3f", annot_kws={"size":7})
    ax.set_xlabel("Rollout Horizon", fontsize=11)
    ax.set_ylabel("Attack Class", fontsize=11)
    ax.set_title("Mean P(class) at Each Rollout Step — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_feature_saliency(saliency_atk, saliency_ben, feature_cols, eng_features, path):
    """Top-20 features by gradient saliency for ATTACK vs BENIGN sequences."""
    top_idx = np.argsort(saliency_atk)[-20:][::-1]
    top_feats= [feature_cols[i] for i in top_idx]
    top_atk  = [saliency_atk[i] for i in top_idx]
    top_ben  = [saliency_ben[i] for i in top_idx]
    eng_set  = set(eng_features)
    colors   = ["#e74c3c" if f in eng_set else "#2980b9" for f in top_feats]

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,7))
    ax1.barh(top_feats[::-1], top_atk[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax1.set_title("Top 20 Features — ATTACK Sequences", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Mean |Gradient × Input|", fontsize=10)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    top_ben_sorted = [saliency_ben[i] for i in top_idx]
    ax2.barh(top_feats[::-1], top_ben_sorted[::-1], color="#95a5a6", edgecolor="white", height=0.7)
    ax2.set_title("Top 20 Features — BENIGN Sequences", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Mean |Gradient × Input|", fontsize=10)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="#e74c3c",label="Engineered"),
                         Patch(facecolor="#2980b9",label="Original")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5,0))
    plt.suptitle("Gradient Saliency Explainability — ATTACK vs BENIGN", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0,0.05,1,1])
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


def plot_mitre_next_stage(sequences_info, path):
    """Bar chart of predicted MITRE ATT&CK next stage distribution."""
    from collections import Counter
    stage_counts = Counter()
    for sinfo in sequences_info:
        k1_probs = sinfo["class_probs"][0]   # k=1 step prediction
        top_cls  = int(np.argmax(k1_probs))
        stage    = MITRE_MAPPING[top_cls]["stage"]
        tactic   = MITRE_MAPPING[top_cls]["tactic"] or "BENIGN"
        stage_counts[tactic] += 1

    tactics  = list(stage_counts.keys())
    counts   = [stage_counts[t] for t in tactics]
    colors   = [TACTIC_COLORS.get(t,"#95a5a6") for t in tactics]
    order    = sorted(range(len(counts)), key=lambda i:-counts[i])
    tactics  = [tactics[i] for i in order]
    counts   = [counts[i]  for i in order]
    colors   = [colors[i]  for i in order]

    fig, ax = plt.subplots(figsize=(10,5))
    bars = ax.barh(tactics[::-1], counts[::-1], color=colors[::-1], edgecolor="white", height=0.6)
    ax.set_xlabel("Number of Sequences", fontsize=11)
    ax.set_title("Predicted Next MITRE ATT&CK Tactic (k=1 step)", fontsize=13, fontweight="bold")
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width()+5, bar.get_y()+bar.get_height()/2,
                f"{count:,}", va="center", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 10 – FORECASTING ENGINE (ROLLOUT + MITRE + RISK + EXPLAINABILITY)")
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
    print("\n[STEP 1] Loading definitions and GRU World Model...")
    with open(PROC_DIR/"feature_list.json")   as f: feat_def  = json.load(f)
    with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)
    with open(MODELS_DIR/"standard_scaler.pkl","rb") as f:
        scaler = pickle.load(f)["scaler"]

    FEATURE_COLS = feat_def["all_numeric_features"]
    CLASS_NAMES  = label_enc["classes"]
    ENG_FEATS    = feat_def["engineered_features"]

    # Load GRU checkpoint (Phase 8)
    ckpt = torch.load(WM_DIR/"gru_world_model.pt", map_location=device)
    hp   = ckpt["hyperparams"]
    model = GRUWorldModel(
        input_dim    = hp["input_dim"],
        hidden_dim   = hp["hidden_dim"],
        num_layers   = hp["num_layers"],
        n_classes_mc = hp["n_classes_mc"],
        dropout      = hp["dropout"]
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  GRU loaded: {n_params:,} params  best_epoch={ckpt['best_epoch']}")

    # ── Load test dataset ─────────────────────────────────────────────────
    print("\n[STEP 2] Loading test sequences...")
    df_test = pd.read_parquet(PROC_DIR/"state_test.parquet")
    df_test = df_test.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    test_ds = NetworkSequenceDataset(df_test, SEQ_LEN, FEATURE_COLS, scaler=scaler, stride=1)
    print(f"  Test sequences: {len(test_ds):,}")

    # Sample N_EVAL for rollout (full rollout is expensive)
    rng = np.random.default_rng(SEED)
    eval_indices = rng.choice(len(test_ds), size=min(N_EVAL, len(test_ds)), replace=False)
    eval_indices = sorted(eval_indices)

    # ── Engines ───────────────────────────────────────────────────────────
    rollout_engine = RolloutEngine(model, device, seq_len=SEQ_LEN)
    risk_scorer    = RiskScorer(discount=RISK_DISCOUNT)
    grad_explainer = GradientSaliency(model, device)

    # ── STEP 3: K-step Rollout on test sequences ─────────────────────────
    print(f"\n[STEP 3] Running {ROLLOUT_K}-step rollout on {len(eval_indices)} test sequences...")
    all_attack_probs = np.zeros((len(eval_indices), ROLLOUT_K))   # (N, K)
    all_class_probs  = np.zeros((len(eval_indices), ROLLOUT_K, N_CLASSES_MC))
    all_true_bins    = []
    all_true_mcs     = []
    sequences_info   = []   # for timeline plots
    risk_scores      = []
    risk_levels      = []

    t_rollout = time.time()
    for i, ds_idx in enumerate(eval_indices):
        X, y_state, y_bin, y_mc = test_ds[ds_idx]
        X_np   = X.numpy()   # (seq_len, D)
        y_b    = int(y_bin.item())
        y_m    = int(y_mc.item())

        atk_probs, cls_probs, _ = rollout_engine.rollout(X_np, K=ROLLOUT_K)

        all_attack_probs[i] = atk_probs
        all_class_probs[i]  = cls_probs
        all_true_bins.append(y_b)
        all_true_mcs.append(y_m)

        rs    = risk_scorer.score(atk_probs, cls_probs)
        rl    = risk_scorer.classify_risk(rs)
        risk_scores.append(rs)
        risk_levels.append(rl)
        sequences_info.append({
            "attack_probs": atk_probs,
            "class_probs":  cls_probs,
            "true_label_binary": y_b,
            "true_label_mc":     y_m,
            "risk_score":  rs,
            "risk_level":  rl
        })

        if (i+1) % 500 == 0:
            elapsed_r = time.time()-t_rollout
            print(f"  [{i+1}/{len(eval_indices)}] rollout done  ({elapsed_r:.1f}s)")

    rollout_time = round(time.time()-t_rollout, 2)
    print(f"  Rollout complete in {rollout_time}s  ({rollout_time/len(eval_indices)*1000:.1f}ms/seq)")

    all_true_bins = np.array(all_true_bins)
    all_true_mcs  = np.array(all_true_mcs)
    risk_scores   = np.array(risk_scores)

    # ── STEP 4: Rollout AUC at each horizon ──────────────────────────────
    print(f"\n[STEP 4] Computing rollout AUC at each horizon k=1..{ROLLOUT_K}...")
    rollout_aucs = []
    rollout_aps  = []
    has_both_classes = len(np.unique(all_true_bins)) == 2

    for k in range(ROLLOUT_K):
        if has_both_classes:
            auc = float(roc_auc_score(all_true_bins, all_attack_probs[:, k]))
            ap  = float(average_precision_score(all_true_bins, all_attack_probs[:, k]))
        else:
            auc = 0.0; ap = 0.0
        rollout_aucs.append(auc)
        rollout_aps.append(ap)
        pred_bin = (all_attack_probs[:, k] >= 0.5).astype(int)
        f1  = float(f1_score(all_true_bins, pred_bin, average="macro", zero_division=0))
        rec = float(recall_score(all_true_bins, pred_bin, average="macro", zero_division=0))
        print(f"  k={k+1}  AUC={auc:.4f}  AP={ap:.4f}  F1={f1:.4f}  Recall={rec:.4f}")

    # Single-step direct prediction (baseline)
    print("\n  Single-step direct prediction (baseline):")
    test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)
    all_y_b_all, all_p_b_all = [], []
    model.eval()
    with torch.no_grad():
        for X_b, _, y_b_b, _ in test_loader:
            _, pb, _ = model(X_b.to(device))
            all_y_b_all.extend(y_b_b.numpy())
            all_p_b_all.extend(F.softmax(pb,dim=1)[:,1].cpu().numpy())
    all_y_b_all = np.array(all_y_b_all); all_p_b_all = np.array(all_p_b_all)
    single_step_auc = float(roc_auc_score(all_y_b_all, all_p_b_all)) if len(np.unique(all_y_b_all))==2 else 0.0
    print(f"  Single-step AUC (full test): {single_step_auc:.4f}")

    # ── STEP 5: Risk Score Analysis ───────────────────────────────────────
    print(f"\n[STEP 5] Risk Score Analysis...")
    from collections import Counter
    level_counts = Counter(risk_levels)
    print(f"  Risk score stats: mean={risk_scores.mean():.2f}  "
          f"std={risk_scores.std():.2f}  "
          f"min={risk_scores.min():.2f}  max={risk_scores.max():.2f}")
    for level in ["SAFE","LOW","MEDIUM","HIGH","CRITICAL"]:
        cnt = level_counts.get(level, 0)
        pct = 100*cnt/len(risk_scores)
        print(f"  {level:<10}: {cnt:>5} ({pct:5.1f}%)")

    # Risk score by true label
    rs_benign = risk_scores[all_true_bins == 0]
    rs_attack = risk_scores[all_true_bins == 1]
    print(f"\n  Mean risk score — BENIGN: {rs_benign.mean():.2f}  ATTACK: {rs_attack.mean():.2f}")

    # ── STEP 6: Gradient Saliency ─────────────────────────────────────────
    print(f"\n[STEP 6] Gradient Saliency Explainability...")
    attack_idxs = np.where(all_true_bins == 1)[0][:200]
    benign_idxs = np.where(all_true_bins == 0)[0][:200]

    X_atk = np.stack([test_ds[eval_indices[i]][0].numpy() for i in attack_idxs])
    X_ben = np.stack([test_ds[eval_indices[i]][0].numpy() for i in benign_idxs])

    model.train()   # enable grad through dropout for saliency
    saliency_atk = grad_explainer.compute(X_atk)
    saliency_ben = grad_explainer.compute(X_ben)
    model.eval()

    top5_atk = sorted(enumerate(saliency_atk), key=lambda x:-x[1])[:10]
    print(f"  Top 10 features driving ATTACK prediction:")
    for feat_idx, sal in top5_atk:
        marker = " [ENG]" if FEATURE_COLS[feat_idx] in set(ENG_FEATS) else ""
        print(f"    {FEATURE_COLS[feat_idx]:<40}: {sal:.4f}{marker}")

    # ── STEP 7: MITRE Stage Forecasting ──────────────────────────────────
    print(f"\n[STEP 7] MITRE ATT&CK Stage Forecasting...")
    # K=1 step predictions → MITRE tactic distribution
    top_classes_k1 = np.argmax(all_class_probs[:, 0, :], axis=1)  # (N,)
    tactic_dist = Counter()
    for tc in top_classes_k1:
        tactic = MITRE_MAPPING[tc]["tactic"] or "BENIGN"
        tactic_dist[tactic] += 1

    print(f"  Predicted MITRE tactic at k=1 (most likely next step):")
    for tactic, cnt in sorted(tactic_dist.items(), key=lambda x:-x[1]):
        pct = 100*cnt/len(top_classes_k1)
        print(f"    {tactic:<30}: {cnt:>5} ({pct:5.1f}%)")

    # Progressive tactic shift across k steps
    print(f"\n  Tactic probability shift across k steps (mean P):")
    unique_tactics = sorted(set(MITRE_MAPPING[i]["tactic"] or "BENIGN" for i in range(N_CLASSES_MC)))
    # Group class probs by tactic
    tactic_probs = {}
    for tactic in unique_tactics:
        class_idxs = [i for i in range(N_CLASSES_MC) if (MITRE_MAPPING[i]["tactic"] or "BENIGN")==tactic]
        tactic_probs[tactic] = all_class_probs[:, :, class_idxs].sum(axis=2).mean(axis=0)  # (K,)

    print(f"  {'Tactic':<28} " + "  ".join(f"k={k+1}" for k in range(ROLLOUT_K)))
    for tactic in unique_tactics:
        vals = " ".join(f"{v:.3f}" for v in tactic_probs[tactic])
        print(f"  {tactic:<28} {vals}")

    # ── STEP 8: Plots ─────────────────────────────────────────────────────
    print(f"\n[STEP 8] Generating plots...")

    # 8a: Rollout AUC
    plot_rollout_auc(rollout_aucs, single_step_auc,
                      PLOTS_DIR/"phase10_rollout_auc.png")
    print("  Rollout AUC plot saved.")

    # 8b: Risk distribution
    plot_risk_distribution(risk_scores.tolist(), risk_levels,
                            PLOTS_DIR/"phase10_risk_distribution.png")
    print("  Risk distribution plot saved.")

    # 8c: Attack progression timelines (6 sample sequences — mix of attack/benign)
    attack_seqs = [s for s in sequences_info if s["true_label_binary"]==1][:3]
    benign_seqs = [s for s in sequences_info if s["true_label_binary"]==0][:3]
    sample_seqs = attack_seqs[:3] + benign_seqs[:3]
    plot_attack_progression_timeline(sample_seqs, CLASS_NAMES,
                                      PLOTS_DIR/"phase10_attack_progression.png")
    print("  Attack progression timeline saved.")

    # 8d: MITRE tactic distribution
    plot_mitre_tactic_distribution(all_class_probs, 
                                    PLOTS_DIR/"phase10_mitre_tactic_dist.png", CLASS_NAMES)
    print("  MITRE tactic distribution heatmap saved.")

    # 8e: Feature saliency
    plot_feature_saliency(saliency_atk, saliency_ben, FEATURE_COLS, ENG_FEATS,
                           PLOTS_DIR/"phase10_feature_saliency.png")
    print("  Feature saliency plot saved.")

    # 8f: MITRE next stage bar
    plot_mitre_next_stage(sequences_info, PLOTS_DIR/"phase10_mitre_next_stage.png")
    print("  MITRE next stage bar saved.")

    # ── STEP 9: Save full metrics ─────────────────────────────────────────
    print(f"\n[STEP 9] Saving metrics...")
    top_features_atk = [{"feature": FEATURE_COLS[i], "saliency": round(float(v),6),
                           "engineered": FEATURE_COLS[i] in set(ENG_FEATS)}
                          for i,v in sorted(enumerate(saliency_atk), key=lambda x:-x[1])[:20]]

    elapsed = round(time.time()-t0, 2)
    final_metrics = {
        "phase": 10, "model": "GRU_WorldModel_Phase8",
        "rollout_k": ROLLOUT_K, "n_eval_sequences": len(eval_indices),
        "risk_discount": RISK_DISCOUNT,
        "rollout_aucs":    [round(v,6) for v in rollout_aucs],
        "rollout_aps":     [round(v,6) for v in rollout_aps],
        "single_step_auc": round(single_step_auc, 6),
        "auc_degradation_k1_to_k8": round(rollout_aucs[0]-rollout_aucs[-1], 4),
        "risk_score_stats": {
            "mean": round(float(risk_scores.mean()),3),
            "std":  round(float(risk_scores.std()),3),
            "min":  round(float(risk_scores.min()),3),
            "max":  round(float(risk_scores.max()),3),
            "mean_benign": round(float(rs_benign.mean()),3) if len(rs_benign)>0 else 0,
            "mean_attack": round(float(rs_attack.mean()),3) if len(rs_attack)>0 else 0,
        },
        "risk_level_counts": dict(level_counts),
        "mitre_tactic_k1": {k:int(v) for k,v in tactic_dist.items()},
        "top20_attack_saliency_features": top_features_atk,
        "elapsed_seconds": elapsed
    }
    mpath = METRICS_DIR/"phase10_forecasting_metrics.json"
    with open(mpath,"w") as f: json.dump(final_metrics, f, indent=2)
    print(f"  Saved: {mpath}")

    # ── Final Summary ─────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"PHASE 10 COMPLETE | Elapsed: {elapsed}s")
    print(f"{'='*72}")
    print(f"\n  ROLLOUT PERFORMANCE (AUC per horizon):")
    for k, auc in enumerate(rollout_aucs):
        bar = "█"*int(auc*40)
        print(f"    k={k+1}  {auc:.4f}  {bar}")
    print(f"\n  AUC degradation k=1→k={ROLLOUT_K}: {final_metrics['auc_degradation_k1_to_k8']:+.4f}")
    print(f"  Single-step AUC (baseline):   {single_step_auc:.4f}")
    print(f"\n  RISK SCORES:  mean={risk_scores.mean():.1f}  BENIGN={rs_benign.mean():.1f}  ATTACK={rs_attack.mean():.1f}")
    print(f"\n  TOP 5 FEATURES (ATTACK saliency):")
    for ft in top_features_atk[:5]:
        eng = " [ENG]" if ft["engineered"] else ""
        print(f"    {ft['feature']:<40}: {ft['saliency']:.4f}{eng}")
    print(f"\n  MITRE ATT&CK k=1 predicted tactic distribution:")
    for tactic, cnt in sorted(tactic_dist.items(), key=lambda x:-x[1])[:5]:
        print(f"    {tactic:<30}: {cnt:,}")
    print(f"\n  PLOTS SAVED (6 plots):")
    for fname in ["phase10_rollout_auc.png","phase10_risk_distribution.png",
                   "phase10_attack_progression.png","phase10_mitre_tactic_dist.png",
                   "phase10_feature_saliency.png","phase10_mitre_next_stage.png"]:
        print(f"    results/plots/{fname}")
    print(f"\n{'='*72}")


if __name__ == "__main__":
    main()
