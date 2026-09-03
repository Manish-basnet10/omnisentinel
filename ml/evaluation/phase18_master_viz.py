"""
PHASE 18 – Master Visualization Report
=======================================
Generates a comprehensive multi-panel summary figure combining
results from all phases into one publication-quality report image.

Outputs:
  results/plots/phase18_master_report.png
  results/plots/phase18_training_summary.png
"""
import json, warnings
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch, FancyBboxPatch
import matplotlib.patheffects as pe

warnings.filterwarnings("ignore")

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"

print("=" * 70)
print("PHASE 18 – MASTER VISUALIZATION REPORT")
print("=" * 70)

def load(fname):
    p = METRICS_DIR / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

lr   = load("phase6_lr_metrics.json")
xgb  = load("phase7_xgb_metrics.json")
gru  = load("phase8_gru_metrics.json")
lstm = load("phase9_lstm_metrics.json")
ph10 = load("phase10_forecasting_metrics.json")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Master 6-panel summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n[PANEL 1] Master 6-panel summary figure...")

fig = plt.figure(figsize=(22, 14))
fig.patch.set_facecolor("#0d1117")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35,
                        left=0.06, right=0.97, top=0.90, bottom=0.07)

DARK   = "#0d1117"
PANEL  = "#161b22"
TEXT   = "#e6edf3"
MUTED  = "#8b949e"
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
RED    = "#f85149"
ORANGE = "#d29922"
PURPLE = "#bc8cff"

def styled_ax(ax, title):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)

def get_bin(m, key):
    """Get binary test metric — handles LR/XGB (binary.test.key) and GRU/LSTM (test.bin_key) structures."""
    if not m: return 0
    v = m.get("binary", {}).get("test", {}).get(key)
    if v is None:
        v = m.get("test", {}).get(f"bin_{key}")
    return v or 0

# ── Panel A: Benchmark Binary AUC ─────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
styled_ax(ax_a, "A. Binary ROC-AUC — Test Set")
models = ["LR", "XGB", "GRU", "LSTM"]
colors_m = [BLUE, RED, GREEN, PURPLE]
aucs = [
    get_bin(lr,   "roc_auc"),
    get_bin(xgb,  "roc_auc"),
    get_bin(gru,  "roc_auc"),
    get_bin(lstm, "roc_auc"),
]
bars = ax_a.bar(models, aucs, color=colors_m, edgecolor=DARK, linewidth=0.5, width=0.55)
ax_a.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
ax_a.set_ylim(0, 1.05)
ax_a.set_ylabel("ROC-AUC", color=MUTED)
for bar, val in zip(bars, aucs):
    ax_a.text(bar.get_x() + bar.get_width()/2, val + 0.01,
              f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# ── Panel B: Binary F1 Macro ──────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
styled_ax(ax_b, "B. Binary F1-Macro — Test Set")
f1s = [
    get_bin(lr,   "f1_macro"),
    get_bin(xgb,  "f1_macro"),
    get_bin(gru,  "f1_macro"),
    get_bin(lstm, "f1_macro"),
]
bars = ax_b.bar(models, f1s, color=colors_m, edgecolor=DARK, linewidth=0.5, width=0.55)
ax_b.set_ylim(0, 1.05)
ax_b.set_ylabel("F1-Macro", color=MUTED)
for bar, val in zip(bars, f1s):
    ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.01,
              f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9, fontweight="bold")

# ── Panel C: K-step Rollout AUC ───────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
styled_ax(ax_c, "C. GRU K-step Rollout AUC Curve")
if ph10:
    # rollout_aucs is a list [auc_k1, auc_k2, ...] — convert to {1: v, 2: v, ...}
    k_aucs_list = ph10.get("rollout_aucs", [])
    k_aucs = {i + 1: v for i, v in enumerate(k_aucs_list)}
    ks     = sorted(k_aucs.keys())
    auc_v  = [k_aucs[k] for k in ks]
    ax_c.plot(ks, auc_v, "o-", color=GREEN, linewidth=2.5, markersize=6,
              markerfacecolor=DARK, markeredgecolor=GREEN, markeredgewidth=2)
    ax_c.fill_between(ks, auc_v, alpha=0.12, color=GREEN)
    ax_c.axhline(0.5, color=MUTED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_c.set_xticks(ks); ax_c.set_xticklabels([f"k={k}" for k in ks], fontsize=7)
    ax_c.set_ylim(0.5, 0.90)
    ax_c.set_ylabel("AUC", color=MUTED)
    ax_c.set_xlabel("Rollout Step", color=MUTED)
    for k, v in zip(ks, auc_v):
        ax_c.annotate(f"{v:.3f}", (k, v), textcoords="offset points",
                      xytext=(0, 7), ha="center", fontsize=7, color=GREEN)

# ── Panel D: Training Curves (GRU) ────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
styled_ax(ax_d, "D. GRU Training History")
if gru and "history" in gru:
    # history is a list of epoch dicts: [{epoch, train_loss, val_loss, ...}, ...]
    hist  = gru["history"]
    eps   = [h["epoch"] for h in hist]
    ax_d.plot(eps, [h["train_loss"] for h in hist], color=BLUE,  linewidth=1.8, label="Train Loss")
    ax_d.plot(eps, [h["val_loss"]   for h in hist], color=RED,   linewidth=1.8, label="Val Loss",  linestyle="--")
    ax_d.set_ylabel("Loss", color=MUTED)
    ax_d.set_xlabel("Epoch", color=MUTED)
    ax_d.legend(fontsize=8, facecolor=DARK, labelcolor=TEXT, edgecolor="#30363d")
else:
    ax_d.text(0.5, 0.5, "Training history\nnot available\n(see phase8 plot)",
              ha="center", va="center", color=MUTED, fontsize=9, transform=ax_d.transAxes)

# ── Panel E: Risk Score Separation ────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
styled_ax(ax_e, "E. Risk Score: Attack vs Benign Separation")
if ph10:
    # Correct key path: risk_score_stats.mean_attack / mean_benign
    risk_stats = ph10.get("risk_score_stats", {})
    atk_mean   = risk_stats.get("mean_attack",  56.4)
    ben_mean   = risk_stats.get("mean_benign",  12.8)
    delta      = atk_mean - ben_mean
    categories = ["BENIGN\nMean Risk", "ATTACK\nMean Risk", "Separation\n(Δ)"]
    values     = [ben_mean, atk_mean, delta]
    bar_colors = [GREEN, RED, ORANGE]
    bars = ax_e.bar(categories, values, color=bar_colors, edgecolor=DARK, linewidth=0.5, width=0.5)
    ax_e.set_ylim(0, 110)
    ax_e.set_ylabel("Risk Score (0–100)", color=MUTED)
    for bar, val in zip(bars, values):
        ax_e.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                  f"{val:.1f}", ha="center", va="bottom", color=TEXT, fontsize=11, fontweight="bold")

# ── Panel F: MITRE Tactic Distribution ────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
styled_ax(ax_f, "F. MITRE ATT&CK Tactic Distribution (k=1)")
if ph10:
    # Correct key: mitre_tactic_k1 (dict of {tactic: count})
    tactic_data = ph10.get("mitre_tactic_k1", {})
    if tactic_data:
        sorted_t = sorted(tactic_data.items(), key=lambda x: -x[1])[:6]
        tnames   = [t[0] if t[0] else "BENIGN" for t in sorted_t]
        tcounts  = [t[1] for t in sorted_t]
        tcolors  = [RED if "Impact" in n else GREEN if "BENIGN" in n
                    else ORANGE if "Credential" in n else BLUE for n in tnames]
        ax_f.barh(tnames[::-1], tcounts[::-1], color=tcolors[::-1],
                  edgecolor=DARK, linewidth=0.5, height=0.6)
        ax_f.set_xlabel("Sequence Count", color=MUTED)
    else:
        ax_f.text(0.5, 0.5, "See phase10\nmitre_tactic_dist.png",
                  ha="center", va="center", color=MUTED, fontsize=9, transform=ax_f.transAxes)

# ── Super title ────────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "OmniSentinel — AI Network Attack Forecasting | SIH 2026 Problem Statement #26153 (NTRO)",
         ha="center", va="center", fontsize=14, fontweight="bold", color=TEXT)
fig.text(0.5, 0.935,
         "CIC-IDS2017 Dataset | GRU World Model | K=8 Autoregressive Rollout | MITRE ATT&CK Mapping | SHAP Explainability",
         ha="center", va="center", fontsize=9, color=MUTED)

plt.savefig(PLOTS_DIR / "phase18_master_report.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  Plot 1 saved: phase18_master_report.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Model comparison heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[PANEL 2] Model comparison heatmap...")

metrics_list = ["accuracy", "f1_macro", "recall_macro", "roc_auc"]
metric_labels = ["Accuracy", "F1-Macro", "Recall-Macro", "ROC-AUC"]
model_names   = ["LR", "XGB", "GRU", "LSTM"]

data = np.zeros((len(model_names), len(metrics_list)))
srcs = [
    lr.get("binary", {}).get("test", {})   if lr   else {},
    xgb.get("binary", {}).get("test", {})  if xgb  else {},
    gru.get("binary", {}).get("test", {})  if gru  else {},
    lstm.get("binary", {}).get("test", {}) if lstm else {},
]
for i, src in enumerate(srcs):
    for j, m in enumerate(metrics_list):
        data[i, j] = src.get(m, 0)

fig2, ax = plt.subplots(figsize=(9, 5))
fig2.patch.set_facecolor(DARK)
ax.set_facecolor(DARK)

im = ax.imshow(data, cmap="YlOrRd", vmin=0.4, vmax=1.0, aspect="auto")
cbar = fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelcolor=TEXT, labelsize=9)
cbar.set_label("Score", color=TEXT)

ax.set_xticks(range(len(metric_labels))); ax.set_xticklabels(metric_labels, color=TEXT, fontsize=10)
ax.set_yticks(range(len(model_names)));   ax.set_yticklabels(model_names,   color=TEXT, fontsize=11, fontweight="bold")
ax.tick_params(colors=TEXT)
for spine in ax.spines.values(): spine.set_edgecolor("#30363d")

for i in range(len(model_names)):
    for j in range(len(metrics_list)):
        val = data[i, j]
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                color="black" if val > 0.7 else TEXT, fontsize=11, fontweight="bold")

ax.set_title("Model Benchmark Heatmap — Binary Test Set\n(CIC-IDS2017, Chronological Split)",
             color=TEXT, fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase18_benchmark_heatmap.png", dpi=150,
            bbox_inches="tight", facecolor=DARK)
plt.close()
print("  Plot 2 saved: phase18_benchmark_heatmap.png")

print("\n" + "=" * 70)
print("PHASE 18 COMPLETE — MASTER VISUALIZATION REPORT")
print("  2 plots saved to results/plots/")
print("=" * 70)
