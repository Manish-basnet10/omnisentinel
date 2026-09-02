"""
PHASE 6 – Logistic Regression Baseline
========================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

MANDATORY RULES:
  - StandardScaler fit ONLY on training data
  - Train/Val/Test strictly separated
  - ALL metrics come from real predictions — zero fabrication
  - Fixed seed = 42
  - Save model, scaler, feature list, metrics

APPROACH:
  Task A — Binary classification    (0=BENIGN, 1=ATTACK)
  Task B — Multiclass classification (15 classes)

  Training data: full train set (1,926,056 rows).
    Solver: 'saga' (designed for large datasets, supports L1/L2/elasticnet).
    If saga times out in max_iter=200, we report partial convergence honestly.

  Class imbalance: class_weight='balanced' (mandatory for fair comparison)

OUTPUTS:
  models/lr_binary.pkl         — binary LR model
  models/lr_multiclass.pkl     — multiclass LR model
  models/standard_scaler.pkl   — canonical scaler (reused by GRU/LSTM)
  results/metrics/phase6_lr_metrics.json
  results/plots/phase6_confusion_matrix_binary.png
  results/plots/phase6_confusion_matrix_multiclass.png
"""

import os, json, warnings, time, gc
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, average_precision_score
)

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PLOTS_DIR   = BASE_DIR / "results" / "plots"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, task="binary", class_names=None):
    """Compute a comprehensive metric dict. All values from real predictions."""
    m = {}
    m["accuracy"]            = float(accuracy_score(y_true, y_pred))
    m["precision_macro"]     = float(precision_score(y_true, y_pred, average="macro",   zero_division=0))
    m["precision_weighted"]  = float(precision_score(y_true, y_pred, average="weighted",zero_division=0))
    m["recall_macro"]        = float(recall_score(y_true, y_pred,    average="macro",   zero_division=0))
    m["recall_weighted"]     = float(recall_score(y_true, y_pred,    average="weighted",zero_division=0))
    m["f1_macro"]            = float(f1_score(y_true, y_pred,        average="macro",   zero_division=0))
    m["f1_weighted"]         = float(f1_score(y_true, y_pred,        average="weighted",zero_division=0))

    if task == "binary" and y_prob is not None:
        m["roc_auc"]         = float(roc_auc_score(y_true, y_prob[:, 1]))
        m["avg_precision"]   = float(average_precision_score(y_true, y_prob[:, 1]))

    if task == "multiclass" and y_prob is not None:
        try:
            m["roc_auc_macro_ovr"] = float(roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro"))
        except Exception as e:
            m["roc_auc_macro_ovr"] = f"ERROR: {e}"

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels_present = sorted(set(y_true))
    if class_names:
        m["per_class_f1"] = {class_names[i]: round(float(v), 4)
                              for i, v in zip(labels_present, per_class_f1)}
    else:
        m["per_class_f1"] = {str(i): round(float(v), 4)
                              for i, v in zip(labels_present, per_class_f1)}
    return m


def save_confusion_matrix(y_true, y_pred, labels, title, path, figsize=(10,8)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=[str(l) for l in labels])
    disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved confusion matrix: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 6 – LOGISTIC REGRESSION BASELINE")
    print("="*72)

    report = {"phase": 6, "model": "LogisticRegression", "seed": SEED, "steps": []}

    # ── STEP 1: Load feature list and label encoding ───────────────────────
    print("\n[STEP 1] Loading feature definitions...")
    with open(PROC_DIR / "feature_list.json")    as f: feat_def  = json.load(f)
    with open(PROC_DIR / "label_encoding.json")  as f: label_enc = json.load(f)

    FEATURE_COLS = feat_def["all_numeric_features"]   # 60 features
    CLASS_NAMES  = label_enc["classes"]               # 15 classes (sorted)
    D = len(FEATURE_COLS)
    print(f"  Features: {D}  |  Classes: {len(CLASS_NAMES)}")

    # ── STEP 2: Load train, val, test splits ─────────────────────────────
    print("\n[STEP 2] Loading state splits...")
    df_train = pd.read_parquet(PROC_DIR / "state_train.parquet")
    df_val   = pd.read_parquet(PROC_DIR / "state_val.parquet")
    df_test  = pd.read_parquet(PROC_DIR / "state_test.parquet")
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # Extract arrays
    X_train = df_train[FEATURE_COLS].values.astype(np.float32)
    y_train_bin = df_train["label_binary"].values.astype(int)
    y_train_mc  = df_train["label_multiclass"].values.astype(int)

    X_val   = df_val[FEATURE_COLS].values.astype(np.float32)
    y_val_bin = df_val["label_binary"].values.astype(int)
    y_val_mc  = df_val["label_multiclass"].values.astype(int)

    X_test  = df_test[FEATURE_COLS].values.astype(np.float32)
    y_test_bin = df_test["label_binary"].values.astype(int)
    y_test_mc  = df_test["label_multiclass"].values.astype(int)

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    print(f"  Train binary: {np.bincount(y_train_bin).tolist()}")
    print(f"  Val   binary: {np.bincount(y_val_bin).tolist()}")
    print(f"  Test  binary: {np.bincount(y_test_bin).tolist()}")
    report["steps"].append({"step":2,"n_train":len(df_train),
                             "n_val":len(df_val),"n_test":len(df_test)})
    del df_train, df_val, df_test; gc.collect()

    # ── STEP 3: Fit StandardScaler on TRAINING data ONLY ─────────────────
    print("\n[STEP 3] Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    t_sc = time.time()
    scaler.fit(X_train)
    print(f"  Scaler fit in {time.time()-t_sc:.2f}s")

    X_train_sc = scaler.transform(X_train).astype(np.float32)
    X_val_sc   = scaler.transform(X_val).astype(np.float32)
    X_test_sc  = scaler.transform(X_test).astype(np.float32)

    scaler_path = MODELS_DIR / "standard_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "feature_cols": FEATURE_COLS}, f)
    print(f"  Canonical scaler saved: {scaler_path}")
    print(f"  Feature means (first 5): {scaler.mean_[:5].round(4).tolist()}")
    report["steps"].append({"step":3,"scaler":"StandardScaler","fit_on":"train_only"})

    # ── STEP 4A: Binary Logistic Regression ──────────────────────────────
    print("\n[STEP 4A] Training Binary Logistic Regression (BENIGN vs ATTACK)...")
    print(f"  Solver: saga  |  max_iter: 200  |  class_weight: balanced  |  C: 1.0")
    print(f"  Training on {X_train_sc.shape[0]:,} samples × {X_train_sc.shape[1]} features...")

    lr_bin = LogisticRegression(
        solver="saga", C=1.0, max_iter=200,
        class_weight="balanced", random_state=SEED,
        n_jobs=-1, verbose=0, tol=1e-4
    )
    t_fit = time.time()
    lr_bin.fit(X_train_sc, y_train_bin)
    fit_time_bin = round(time.time() - t_fit, 2)
    print(f"  Fit time: {fit_time_bin}s  |  Converged: {lr_bin.n_iter_[0] < 200}")
    print(f"  Iterations used: {lr_bin.n_iter_[0]}")

    # Evaluate binary
    print("\n  Evaluating on Val...")
    y_val_pred_bin  = lr_bin.predict(X_val_sc)
    y_val_prob_bin  = lr_bin.predict_proba(X_val_sc)
    metrics_val_bin = compute_metrics(y_val_bin, y_val_pred_bin, y_val_prob_bin,
                                       task="binary", class_names=["BENIGN","ATTACK"])

    print("\n  Evaluating on Test...")
    y_test_pred_bin  = lr_bin.predict(X_test_sc)
    y_test_prob_bin  = lr_bin.predict_proba(X_test_sc)
    metrics_test_bin = compute_metrics(y_test_bin, y_test_pred_bin, y_test_prob_bin,
                                        task="binary", class_names=["BENIGN","ATTACK"])

    print(f"\n  Binary — VAL  results:")
    for k, v in metrics_val_bin.items():
        if k != "per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"\n  Binary — TEST results:")
    for k, v in metrics_test_bin.items():
        if k != "per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Classification report
    print("\n  Classification Report (Test — Binary):")
    print(classification_report(y_test_bin, y_test_pred_bin,
                                 target_names=["BENIGN","ATTACK"], zero_division=0))

    # Save binary model
    lr_bin_path = MODELS_DIR / "lr_binary.pkl"
    with open(lr_bin_path, "wb") as f:
        pickle.dump(lr_bin, f)
    print(f"  Model saved: {lr_bin_path}")

    # Confusion matrix binary (test)
    cm_bin_path = PLOTS_DIR / "phase6_confusion_matrix_binary.png"
    save_confusion_matrix(y_test_bin, y_test_pred_bin, labels=[0,1],
                           title="LR Binary — Confusion Matrix (Test)\nBENIGN=0  ATTACK=1",
                           path=cm_bin_path, figsize=(6,5))
    report["steps"].append({"step":"4A","task":"binary_lr","fit_time_s":fit_time_bin,
                             "n_iter":int(lr_bin.n_iter_[0])})

    # ── STEP 4B: Multiclass Logistic Regression ───────────────────────────
    print("\n[STEP 4B] Training Multiclass Logistic Regression (15 classes)...")
    print(f"  Solver: saga  |  max_iter: 200  |  class_weight: balanced  |  multi_class: multinomial")

    lr_mc = LogisticRegression(
        solver="saga", C=1.0, max_iter=200,
        class_weight="balanced", random_state=SEED,
        multi_class="multinomial", n_jobs=-1, verbose=0, tol=1e-4
    )
    t_fit_mc = time.time()
    lr_mc.fit(X_train_sc, y_train_mc)
    fit_time_mc = round(time.time() - t_fit_mc, 2)
    print(f"  Fit time: {fit_time_mc}s  |  Converged: {(lr_mc.n_iter_[0] < 200)}")
    print(f"  Iterations used: {lr_mc.n_iter_[0]}")

    # Evaluate multiclass — val
    print("\n  Evaluating on Val...")
    y_val_pred_mc  = lr_mc.predict(X_val_sc)
    y_val_prob_mc  = lr_mc.predict_proba(X_val_sc)
    metrics_val_mc = compute_metrics(y_val_mc, y_val_pred_mc, y_val_prob_mc,
                                      task="multiclass", class_names=CLASS_NAMES)

    # Evaluate multiclass — test
    print("\n  Evaluating on Test...")
    y_test_pred_mc  = lr_mc.predict(X_test_sc)
    y_test_prob_mc  = lr_mc.predict_proba(X_test_sc)
    metrics_test_mc = compute_metrics(y_test_mc, y_test_pred_mc, y_test_prob_mc,
                                       task="multiclass", class_names=CLASS_NAMES)

    print(f"\n  Multiclass — VAL  results:")
    for k, v in metrics_val_mc.items():
        if k != "per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\n  Multiclass — TEST results:")
    for k, v in metrics_test_mc.items():
        if k != "per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print(f"\n  Per-class F1 (Test):")
    for cls, f1 in metrics_test_mc.get("per_class_f1", {}).items():
        print(f"    {cls:<35}: {f1:.4f}")

    print("\n  Classification Report (Test — Multiclass):")
    # Map integer predictions back to class names
    idx2class = label_enc["idx2class"]
    y_test_pred_names = [idx2class[str(p)] for p in y_test_pred_mc]
    y_test_true_names = [idx2class[str(t)] for t in y_test_mc]
    present_classes = sorted(set(y_test_true_names) | set(y_test_pred_names))
    print(classification_report(y_test_true_names, y_test_pred_names,
                                  labels=present_classes, zero_division=0))

    # Save multiclass model
    lr_mc_path = MODELS_DIR / "lr_multiclass.pkl"
    with open(lr_mc_path, "wb") as f:
        pickle.dump(lr_mc, f)
    print(f"  Model saved: {lr_mc_path}")

    # Confusion matrix multiclass (test — classes present only)
    present_label_ints = sorted(set(y_test_mc))
    present_class_names = [CLASS_NAMES[i] for i in present_label_ints]
    cm_mc_path = PLOTS_DIR / "phase6_confusion_matrix_multiclass.png"
    fig, ax = plt.subplots(figsize=(10, 8))
    cm_mc = confusion_matrix(y_test_mc, y_test_pred_mc, labels=present_label_ints)
    sns.heatmap(cm_mc, annot=True, fmt="d", cmap="Blues",
                xticklabels=present_class_names,
                yticklabels=present_class_names, ax=ax, linewidths=0.5)
    ax.set_title("LR Multiclass — Confusion Matrix (Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_mc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved: {cm_mc_path}")

    report["steps"].append({"step":"4B","task":"multiclass_lr","fit_time_s":fit_time_mc,
                             "n_iter":int(lr_mc.n_iter_[0])})

    # ── STEP 5: Feature importance (coefficient magnitudes) ───────────────
    print("\n[STEP 5] Feature importance (coefficient magnitudes — Binary LR)...")
    coef_abs = np.abs(lr_bin.coef_[0])
    feat_importance = sorted(zip(FEATURE_COLS, coef_abs), key=lambda x: -x[1])
    print(f"  Top 15 features by |coefficient|:")
    for feat, imp in feat_importance[:15]:
        print(f"    {feat:<40}: {imp:.4f}")

    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    top_feats  = [f[0] for f in feat_importance[:20]]
    top_imps   = [f[1] for f in feat_importance[:20]]
    colors = ["#e74c3c" if "ENG" in f or any(k in f for k in
              ["ratio","density","log1p","total_"]) else "#3498db" for f in top_feats]
    bars = ax.barh(top_feats[::-1], top_imps[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_xlabel("|Coefficient| (LogisticRegression Binary)", fontsize=11)
    ax.set_title("LR Baseline — Top 20 Feature Importances (Binary)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#e74c3c", label="Engineered"),
                       Patch(facecolor="#3498db", label="Original")]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    fi_path = PLOTS_DIR / "phase6_feature_importance.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Feature importance plot saved: {fi_path}")

    # ── STEP 6: Compile and save final metrics ────────────────────────────
    print("\n[STEP 6] Saving metrics...")
    elapsed = round(time.time() - t0, 2)

    final_metrics = {
        "phase": 6,
        "model": "LogisticRegression",
        "solver": "saga",
        "C": 1.0,
        "max_iter": 200,
        "class_weight": "balanced",
        "seed": SEED,
        "feature_dim": D,
        "n_train": int(len(X_train_sc)),
        "n_val": int(len(X_val_sc)),
        "n_test": int(len(X_test_sc)),
        "scaler": "StandardScaler (fit on train only)",
        "binary": {
            "n_iter": int(lr_bin.n_iter_[0]),
            "converged": bool(lr_bin.n_iter_[0] < 200),
            "fit_time_s": fit_time_bin,
            "val":  {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in metrics_val_bin.items()},
            "test": {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in metrics_test_bin.items()}
        },
        "multiclass": {
            "n_iter": int(lr_mc.n_iter_[0]),
            "converged": bool(lr_mc.n_iter_[0] < 200),
            "fit_time_s": fit_time_mc,
            "val":  {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in metrics_val_mc.items()},
            "test": {k: round(v, 6) if isinstance(v, float) else v
                     for k, v in metrics_test_mc.items()}
        },
        "top_features_binary": [{"feature": f, "coef_abs": round(float(c), 6)}
                                  for f, c in feat_importance[:20]],
        "elapsed_seconds": elapsed
    }

    metrics_path = METRICS_DIR / "phase6_lr_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ── STEP 7: Final summary table ───────────────────────────────────────
    print(f"\n{'='*72}")
    print("PHASE 6 COMPLETE — LOGISTIC REGRESSION BASELINE")
    print(f"{'='*72}")
    print(f"\n  BINARY  (BENIGN vs ATTACK):")
    b = final_metrics["binary"]
    print(f"    {'Metric':<25} {'Val':>10} {'Test':>10}")
    print(f"    {'-'*47}")
    for metric in ["accuracy","f1_macro","f1_weighted","precision_macro","recall_macro","roc_auc"]:
        vv = b["val"].get(metric, "—")
        tv = b["test"].get(metric, "—")
        print(f"    {metric:<25} {vv:>10.4f} {tv:>10.4f}")

    print(f"\n  MULTICLASS (15 classes):")
    m = final_metrics["multiclass"]
    print(f"    {'Metric':<25} {'Val':>10} {'Test':>10}")
    print(f"    {'-'*47}")
    for metric in ["accuracy","f1_macro","f1_weighted","precision_macro","recall_macro"]:
        vv = m["val"].get(metric, "—")
        tv = m["test"].get(metric, "—")
        print(f"    {metric:<25} {vv:>10.4f} {tv:>10.4f}")

    print(f"\n  Elapsed total: {elapsed}s")
    print(f"  Scaler:        {scaler_path.name}")
    print(f"  Models:        lr_binary.pkl  |  lr_multiclass.pkl")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
