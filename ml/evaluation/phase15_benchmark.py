"""
PHASE 15 – Final Benchmark Table (Real Metrics Only)
=====================================================
Reads all phase metric JSONs and produces a single consolidated
benchmark table comparing LR vs XGB vs GRU vs LSTM on the same test set.

Outputs:
  results/metrics/phase15_benchmark_table.json
  results/plots/phase15_benchmark_table.png
"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"

print("=" * 70)
print("PHASE 15 – FINAL BENCHMARK TABLE")
print("=" * 70)

def load(fname):
    with open(METRICS_DIR / fname) as f: return json.load(f)

lr   = load("phase6_lr_metrics.json")
xgb  = load("phase7_xgb_metrics.json")
gru  = load("phase8_gru_metrics.json")
lstm = load("phase9_lstm_metrics.json")

def get_metric(m_dict, section, metric):
    """
    Fetch a test metric from a model dict, handling two JSON structures:

    Structure A — LR / XGB (phase6/7):
        model["binary"]["test"]["roc_auc"]
        model["multiclass"]["test"]["f1_macro"]

    Structure B — GRU / LSTM (phase8/9):
        model["test"]["bin_roc_auc"]
        model["test"]["mc_f1_macro"]
        (no 'binary' or 'multiclass' sub-key — metrics stored flat with prefix)
    """
    if not m_dict:
        return None
    prefix = "bin_" if section == "binary" else "mc_"

    # Try Structure A (LR/XGB): model[section]["test"][metric]
    val = m_dict.get(section, {}).get("test", {}).get(metric)
    if val is not None:
        return val

    # Try Structure B (GRU/LSTM): model["test"][prefix + metric]
    val = m_dict.get("test", {}).get(f"{prefix}{metric}")
    return val

# ── Binary Test Metrics ────────────────────────────────────────────────────────
bin_metrics = ["accuracy", "f1_macro", "f1_weighted", "recall_macro", "roc_auc"]
bin_table = {}
for m in bin_metrics:
    bin_table[m] = {
        "LR":   get_metric(lr,   "binary", m),
        "XGB":  get_metric(xgb,  "binary", m),
        "GRU":  get_metric(gru,  "binary", m),
        "LSTM": get_metric(lstm, "binary", m),
    }

# ── Multiclass Test Metrics ────────────────────────────────────────────────────
mc_metrics = ["accuracy", "f1_macro", "f1_weighted", "recall_macro"]
mc_table = {}
for m in mc_metrics:
    mc_table[m] = {
        "LR":   get_metric(lr,   "multiclass", m),
        "XGB":  get_metric(xgb,  "multiclass", m),
        "GRU":  get_metric(gru,  "multiclass", m),
        "LSTM": get_metric(lstm, "multiclass", m),
    }


# ── Print Table ───────────────────────────────────────────────────────────────
print(f"\n{'BINARY TEST SET':^70}")
print("-" * 70)
print(f"{'Metric':<25} {'LR':>10} {'XGB':>10} {'GRU':>10} {'LSTM':>10}")
print("-" * 70)
for m, vals in bin_table.items():
    row = f"{m:<25}"
    for model in ["LR", "XGB", "GRU", "LSTM"]:
        v = vals.get(model)
        row += f" {v:>10.4f}" if v is not None else f" {'N/A':>10}"
    print(row)

print(f"\n{'MULTICLASS TEST SET':^70}")
print("-" * 70)
print(f"{'Metric':<25} {'LR':>10} {'XGB':>10} {'GRU':>10} {'LSTM':>10}")
print("-" * 70)
for m, vals in mc_table.items():
    row = f"{m:<25}"
    for model in ["LR", "XGB", "GRU", "LSTM"]:
        v = vals.get(model)
        row += f" {v:>10.4f}" if v is not None else f" {'N/A':>10}"
    print(row)

# ── PLOT: Grouped bar chart ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
models = ["LR", "XGB", "GRU", "LSTM"]
colors = ["#3498db", "#e74c3c", "#27ae60", "#9b59b6"]
x = np.arange(len(bin_metrics))
w = 0.18

ax = axes[0]
for i, (model, color) in enumerate(zip(models, colors)):
    vals = [bin_table[m].get(model) or 0 for m in bin_metrics]
    ax.bar(x + i*w, vals, width=w, label=model, color=color, alpha=0.88, edgecolor="white")
ax.set_xticks(x + w*1.5); ax.set_xticklabels(bin_metrics, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.1); ax.set_title("Binary Classification — Test Set", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

ax = axes[1]
x = np.arange(len(mc_metrics))
for i, (model, color) in enumerate(zip(models, colors)):
    vals = [mc_table[m].get(model) or 0 for m in mc_metrics]
    ax.bar(x + i*w, vals, width=w, label=model, color=color, alpha=0.88, edgecolor="white")
ax.set_xticks(x + w*1.5); ax.set_xticklabels(mc_metrics, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.1); ax.set_title("Multiclass Classification — Test Set", fontweight="bold")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.suptitle("Final Model Benchmark: LR vs XGBoost vs GRU World Model vs LSTM\nCIC-IDS2017 Test Set — Real Metrics Only",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "phase15_benchmark_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved: phase15_benchmark_table.png")

# ── Save JSON ──────────────────────────────────────────────────────────────────
benchmark = {
    "phase": 15,
    "dataset": "CIC-IDS2017",
    "split": "test (chronological)",
    "models": models,
    "binary_test": bin_table,
    "multiclass_test": mc_table,
    "winner_binary_auc":   max(models, key=lambda m: bin_table["roc_auc"].get(m) or 0),
    "winner_binary_f1":    max(models, key=lambda m: bin_table["f1_macro"].get(m) or 0),
    "winner_mc_f1":        max(models, key=lambda m: mc_table["f1_macro"].get(m) or 0),
}
with open(METRICS_DIR / "phase15_benchmark_table.json", "w") as f:
    json.dump(benchmark, f, indent=2)

print("\n" + "=" * 70)
print("PHASE 15 COMPLETE — BENCHMARK TABLE")
print(f"  Best Binary AUC:    {benchmark['winner_binary_auc']}")
print(f"  Best Binary F1:     {benchmark['winner_binary_f1']}")
print(f"  Best MC F1:         {benchmark['winner_mc_f1']}")
print("=" * 70)
