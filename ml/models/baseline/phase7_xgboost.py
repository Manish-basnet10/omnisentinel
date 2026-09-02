"""
PHASE 7 – XGBoost Static Comparison Model
==========================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

ROLE: Static tabular comparison model (NOT a world model).
      Establishes a strong non-linear baseline for comparison with GRU/LSTM.
      Uses the SAME scaler, train/val/test splits as Phase 6.

APPROACH:
  Task A — Binary classification    (0=BENIGN, 1=ATTACK)
  Task B — Multiclass classification (15 classes)

  XGBoost config:
    tree_method  = 'hist'     (fast histogram-based, handles large data)
    n_estimators = 500        (with early stopping on val, patience=20)
    max_depth    = 6
    learning_rate= 0.1
    subsample    = 0.8
    colsample_bytree = 0.8
    min_child_weight = 5
    scale_pos_weight: auto (binary only) — handles imbalance

STRICT RULES:
  - Scaler loaded from Phase 6 canonical standard_scaler.pkl
  - Train/Val/Test strictly separated — same splits as Phase 6
  - ALL metrics from real predictions — zero fabrication
  - Fixed seed = 42
  - Save models, feature importance, metrics, plots
"""

import os, json, warnings, time, gc, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    average_precision_score
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
    m = {}
    m["accuracy"]           = float(accuracy_score(y_true, y_pred))
    m["precision_macro"]    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    m["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    m["recall_macro"]       = float(recall_score(y_true, y_pred,    average="macro",    zero_division=0))
    m["recall_weighted"]    = float(recall_score(y_true, y_pred,    average="weighted", zero_division=0))
    m["f1_macro"]           = float(f1_score(y_true, y_pred,        average="macro",    zero_division=0))
    m["f1_weighted"]        = float(f1_score(y_true, y_pred,        average="weighted", zero_division=0))

    if task == "binary" and y_prob is not None:
        m["roc_auc"]       = float(roc_auc_score(y_true, y_prob[:, 1]))
        m["avg_precision"] = float(average_precision_score(y_true, y_prob[:, 1]))

    if task == "multiclass" and y_prob is not None:
        try:
            classes_in_true = np.unique(y_true)
            # Use only columns corresponding to classes present in y_true
            m["roc_auc_macro_ovr"] = float(roc_auc_score(
                y_true, y_prob[:, classes_in_true],
                multi_class="ovr", average="macro",
                labels=classes_in_true))
        except Exception as e:
            m["roc_auc_macro_ovr"] = f"skipped: {e}"

    per_class_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels_present = sorted(set(y_true))
    if class_names:
        m["per_class_f1"] = {class_names[i]: round(float(v), 4)
                              for i, v in zip(labels_present, per_class_f1)}
    else:
        m["per_class_f1"] = {str(i): round(float(v), 4)
                              for i, v in zip(labels_present, per_class_f1)}
    return m


def plot_confusion(y_true, y_pred, labels, label_names, title, path, figsize=(10,8)):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names,
                ax=ax, linewidths=0.4)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path.name}")


def plot_training_curves(evals_result, task_name, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics_avail = list(evals_result.get("train", {}).keys())
    for ax_i, metric in enumerate(metrics_avail[:2]):
        ax = axes[ax_i]
        if "train" in evals_result and metric in evals_result["train"]:
            ax.plot(evals_result["train"][metric], label="Train", color="#3498db", lw=1.5)
        if "val" in evals_result and metric in evals_result["val"]:
            ax.plot(evals_result["val"][metric], label="Val",   color="#e74c3c", lw=1.5)
        ax.set_xlabel("Boosting Rounds", fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f"XGBoost {task_name} — {metric}", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 7 – XGBOOST STATIC COMPARISON MODEL")
    print("="*72)

    report = {"phase": 7, "model": "XGBoost", "seed": SEED, "steps": []}

    # ── STEP 1: Load definitions ──────────────────────────────────────────
    print("\n[STEP 1] Loading feature definitions and canonical scaler...")
    with open(PROC_DIR/"feature_list.json")   as f: feat_def  = json.load(f)
    with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)

    FEATURE_COLS = feat_def["all_numeric_features"]
    CLASS_NAMES  = label_enc["classes"]
    D = len(FEATURE_COLS)
    print(f"  Features: {D}  |  Classes: {len(CLASS_NAMES)}")

    # Load canonical scaler from Phase 6
    with open(MODELS_DIR/"standard_scaler.pkl", "rb") as f:
        scaler_pkg = pickle.load(f)
    scaler = scaler_pkg["scaler"]
    print(f"  Canonical scaler loaded: StandardScaler (fit on train in Phase 6)")

    # ── STEP 2: Load and scale data ───────────────────────────────────────
    print("\n[STEP 2] Loading and scaling train/val/test splits...")
    df_train = pd.read_parquet(PROC_DIR/"state_train.parquet")
    df_val   = pd.read_parquet(PROC_DIR/"state_val.parquet")
    df_test  = pd.read_parquet(PROC_DIR/"state_test.parquet")

    X_train_raw = df_train[FEATURE_COLS].values.astype(np.float32)
    X_val_raw   = df_val[FEATURE_COLS].values.astype(np.float32)
    X_test_raw  = df_test[FEATURE_COLS].values.astype(np.float32)

    y_train_bin = df_train["label_binary"].values.astype(int)
    y_val_bin   = df_val["label_binary"].values.astype(int)
    y_test_bin  = df_test["label_binary"].values.astype(int)

    y_train_mc  = df_train["label_multiclass"].values.astype(int)
    y_val_mc    = df_val["label_multiclass"].values.astype(int)
    y_test_mc   = df_test["label_multiclass"].values.astype(int)

    del df_train, df_val, df_test; gc.collect()

    X_train = scaler.transform(X_train_raw).astype(np.float32)
    X_val   = scaler.transform(X_val_raw).astype(np.float32)
    X_test  = scaler.transform(X_test_raw).astype(np.float32)
    del X_train_raw, X_val_raw, X_test_raw; gc.collect()

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    print(f"  Train binary: {np.bincount(y_train_bin).tolist()}")

    # Compute scale_pos_weight for binary imbalance
    neg, pos = np.bincount(y_train_bin)
    spw = round(neg / pos, 2)
    print(f"  Binary scale_pos_weight = {spw}  (neg/pos = {neg}/{pos})")
    report["steps"].append({"step":2,"n_train":len(X_train),"scale_pos_weight":spw})

    # ── STEP 3A: XGBoost Binary ───────────────────────────────────────────
    print("\n[STEP 3A] Training XGBoost Binary (BENIGN vs ATTACK)...")
    xgb_params_bin = dict(
        objective        = "binary:logistic",
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        scale_pos_weight = spw,
        tree_method      = "hist",
        eval_metric      = ["logloss", "auc"],
        early_stopping_rounds = 20,
        random_state     = SEED,
        n_jobs           = -1,
        verbosity        = 0
    )
    print(f"  Params: {xgb_params_bin}")

    xgb_bin = xgb.XGBClassifier(**xgb_params_bin)
    evals_result_bin = {}
    t_fit = time.time()
    xgb_bin.fit(
        X_train, y_train_bin,
        eval_set=[(X_train, y_train_bin), (X_val, y_val_bin)],
        verbose=50
    )
    fit_time_bin = round(time.time()-t_fit, 2)
    best_iter_bin = xgb_bin.best_iteration
    print(f"\n  Binary fit time: {fit_time_bin}s  |  Best iteration: {best_iter_bin}")

    # Evaluate
    print("\n  Evaluating Binary on Val...")
    y_val_pred_bin  = xgb_bin.predict(X_val)
    y_val_prob_bin  = xgb_bin.predict_proba(X_val)
    metrics_val_bin = compute_metrics(y_val_bin, y_val_pred_bin, y_val_prob_bin,
                                       task="binary", class_names=["BENIGN","ATTACK"])

    print("  Evaluating Binary on Test...")
    y_test_pred_bin  = xgb_bin.predict(X_test)
    y_test_prob_bin  = xgb_bin.predict_proba(X_test)
    metrics_test_bin = compute_metrics(y_test_bin, y_test_pred_bin, y_test_prob_bin,
                                        task="binary", class_names=["BENIGN","ATTACK"])

    print(f"\n  Binary VAL:")
    for k,v in metrics_val_bin.items():
        if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")
    print(f"\n  Binary TEST:")
    for k,v in metrics_test_bin.items():
        if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")

    print("\n  Classification Report (Test — Binary):")
    print(classification_report(y_test_bin, y_test_pred_bin,
                                 target_names=["BENIGN","ATTACK"], zero_division=0))

    # Save binary model
    xgb_bin_path = MODELS_DIR / "xgb_binary.pkl"
    with open(xgb_bin_path,"wb") as f: pickle.dump(xgb_bin, f)
    print(f"  Model saved: {xgb_bin_path}")

    # Plot confusion + curves
    plot_confusion(y_test_bin, y_test_pred_bin, labels=[0,1],
                   label_names=["BENIGN","ATTACK"],
                   title="XGBoost Binary — Confusion Matrix (Test)",
                   path=PLOTS_DIR/"phase7_confusion_matrix_binary.png", figsize=(6,5))

    evals = xgb_bin.evals_result()
    plot_training_curves(
        {"train": evals["validation_0"], "val": evals["validation_1"]},
        "Binary", PLOTS_DIR/"phase7_training_curves_binary.png")

    report["steps"].append({"step":"3A","task":"xgb_binary","fit_time_s":fit_time_bin,
                             "best_iter":int(best_iter_bin)})

    # ── STEP 3B: XGBoost Multiclass ──────────────────────────────────────
    print("\n[STEP 3B] Training XGBoost Multiclass (15 classes)...")

    # Remap multiclass labels to be 0-indexed and contiguous
    # (some classes may not appear in train — keep all 15 for correct indexing)
    unique_train_mc = np.unique(y_train_mc)
    n_classes = len(CLASS_NAMES)  # 15
    print(f"  Classes in train: {unique_train_mc.tolist()}  (total model classes: {n_classes})")

    xgb_params_mc = dict(
        objective        = "multi:softprob",
        num_class        = n_classes,
        n_estimators     = 500,
        max_depth        = 6,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        min_child_weight = 5,
        tree_method      = "hist",
        eval_metric      = ["mlogloss", "merror"],
        early_stopping_rounds = 20,
        random_state     = SEED,
        n_jobs           = -1,
        verbosity        = 0
    )

    xgb_mc = xgb.XGBClassifier(**xgb_params_mc)
    t_fit_mc = time.time()
    xgb_mc.fit(
        X_train, y_train_mc,
        eval_set=[(X_train, y_train_mc), (X_val, y_val_mc)],
        verbose=50
    )
    fit_time_mc = round(time.time()-t_fit_mc, 2)
    best_iter_mc = xgb_mc.best_iteration
    print(f"\n  Multiclass fit time: {fit_time_mc}s  |  Best iteration: {best_iter_mc}")

    # Evaluate multiclass val
    print("\n  Evaluating Multiclass on Val...")
    y_val_pred_mc  = xgb_mc.predict(X_val)
    y_val_prob_mc  = xgb_mc.predict_proba(X_val)
    metrics_val_mc = compute_metrics(y_val_mc, y_val_pred_mc, y_val_prob_mc,
                                      task="multiclass", class_names=CLASS_NAMES)

    # Evaluate multiclass test
    print("  Evaluating Multiclass on Test...")
    y_test_pred_mc  = xgb_mc.predict(X_test)
    y_test_prob_mc  = xgb_mc.predict_proba(X_test)
    metrics_test_mc = compute_metrics(y_test_mc, y_test_pred_mc, y_test_prob_mc,
                                       task="multiclass", class_names=CLASS_NAMES)

    print(f"\n  Multiclass VAL:")
    for k,v in metrics_val_mc.items():
        if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")
    print(f"\n  Multiclass TEST:")
    for k,v in metrics_test_mc.items():
        if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")

    print(f"\n  Per-class F1 (Test):")
    for cls, f1v in metrics_test_mc.get("per_class_f1",{}).items():
        cnt = int(np.sum(y_test_mc == label_enc["class2idx"].get(cls, -1)))
        print(f"    {cls:<35}: F1={f1v:.4f}  (test support={cnt})")

    idx2class = label_enc["idx2class"]
    y_test_pred_names = [idx2class[str(p)] for p in y_test_pred_mc]
    y_test_true_names = [idx2class[str(t)] for t in y_test_mc]
    present_classes   = sorted(set(y_test_true_names)|set(y_test_pred_names))
    print("\n  Classification Report (Test — Multiclass):")
    print(classification_report(y_test_true_names, y_test_pred_names,
                                  labels=present_classes, zero_division=0))

    # Save multiclass model
    xgb_mc_path = MODELS_DIR / "xgb_multiclass.pkl"
    with open(xgb_mc_path,"wb") as f: pickle.dump(xgb_mc, f)
    print(f"  Model saved: {xgb_mc_path}")

    present_label_ints = sorted(set(y_test_mc))
    present_names      = [CLASS_NAMES[i] for i in present_label_ints]
    plot_confusion(y_test_mc, y_test_pred_mc,
                   labels=present_label_ints, label_names=present_names,
                   title="XGBoost Multiclass — Confusion Matrix (Test)",
                   path=PLOTS_DIR/"phase7_confusion_matrix_multiclass.png", figsize=(8,7))

    evals_mc = xgb_mc.evals_result()
    plot_training_curves(
        {"train": evals_mc["validation_0"], "val": evals_mc["validation_1"]},
        "Multiclass", PLOTS_DIR/"phase7_training_curves_multiclass.png")

    report["steps"].append({"step":"3B","task":"xgb_multiclass","fit_time_s":fit_time_mc,
                             "best_iter":int(best_iter_mc)})

    # ── STEP 4: Feature importance (gain-based) ───────────────────────────
    print("\n[STEP 4] Feature importance (gain — XGBoost Binary)...")
    fi_gain  = xgb_bin.get_booster().get_score(importance_type="gain")
    fi_cover = xgb_bin.get_booster().get_score(importance_type="cover")
    fi_weight= xgb_bin.get_booster().get_score(importance_type="weight")

    # Map f0, f1, ... back to feature names
    fi_named = {}
    for k, v in fi_gain.items():
        idx = int(k[1:])  # "f12" → 12
        if idx < len(FEATURE_COLS):
            fi_named[FEATURE_COLS[idx]] = v

    fi_sorted = sorted(fi_named.items(), key=lambda x: -x[1])
    print(f"  Top 15 features by gain:")
    for feat, imp in fi_sorted[:15]:
        print(f"    {feat:<40}: {imp:.2f}")

    # Plot feature importance (top 20)
    top_feats = [x[0] for x in fi_sorted[:20]]
    top_vals  = [x[1] for x in fi_sorted[:20]]
    eng_set   = set(feat_def["engineered_features"])
    colors    = ["#e74c3c" if f in eng_set else "#2980b9" for f in top_feats]

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_xlabel("Gain (XGBoost Binary)", fontsize=11)
    ax.set_title("XGBoost — Top 20 Feature Importance (Gain, Binary)", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#e74c3c",label="Engineered"),
                        Patch(facecolor="#2980b9",label="Original")], loc="lower right")
    plt.tight_layout()
    fi_path = PLOTS_DIR / "phase7_feature_importance_gain.png"
    plt.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fi_path.name}")

    # ── STEP 5: LR vs XGB comparison table ───────────────────────────────
    print("\n[STEP 5] LR vs XGBoost comparison (Test set):")
    try:
        with open(METRICS_DIR/"phase6_lr_metrics.json") as f:
            lr_m = json.load(f)
        print(f"\n  BINARY (Test):")
        print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
        print(f"    {'-'*57}")
        for metric in ["accuracy","f1_macro","f1_weighted","recall_macro","roc_auc"]:
            lr_v  = lr_m["binary"]["test"].get(metric, None)
            xgb_v = metrics_test_bin.get(metric, None)
            if lr_v is not None and xgb_v is not None:
                delta = xgb_v - lr_v
                arrow = "▲" if delta > 0 else "▼"
                print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {arrow}{abs(delta):>9.4f}")

        print(f"\n  MULTICLASS (Test):")
        print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
        print(f"    {'-'*57}")
        for metric in ["accuracy","f1_macro","f1_weighted","recall_macro"]:
            lr_v  = lr_m["multiclass"]["test"].get(metric, None)
            xgb_v = metrics_test_mc.get(metric, None)
            if lr_v is not None and xgb_v is not None:
                delta = xgb_v - lr_v
                arrow = "▲" if delta > 0 else "▼"
                print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {arrow}{abs(delta):>9.4f}")
    except FileNotFoundError:
        print("  (Phase 6 metrics not found — skipping comparison)")

    # ── STEP 6: Save metrics ──────────────────────────────────────────────
    elapsed = round(time.time()-t0, 2)
    final_metrics = {
        "phase": 7, "model": "XGBoost", "seed": SEED,
        "xgb_version": xgb.__version__,
        "binary_params": xgb_params_bin,
        "multiclass_params": xgb_params_mc,
        "binary": {
            "best_iteration": int(best_iter_bin),
            "fit_time_s": fit_time_bin,
            "val":  {k: round(v,6) if isinstance(v,float) else v for k,v in metrics_val_bin.items()},
            "test": {k: round(v,6) if isinstance(v,float) else v for k,v in metrics_test_bin.items()}
        },
        "multiclass": {
            "best_iteration": int(best_iter_mc),
            "fit_time_s": fit_time_mc,
            "val":  {k: round(v,6) if isinstance(v,float) else v for k,v in metrics_val_mc.items()},
            "test": {k: round(v,6) if isinstance(v,float) else v for k,v in metrics_test_mc.items()}
        },
        "top20_features_gain": [{"feature":f,"gain":round(float(g),4)} for f,g in fi_sorted[:20]],
        "elapsed_seconds": elapsed
    }
    mpath = METRICS_DIR / "phase7_xgb_metrics.json"
    with open(mpath,"w") as f: json.dump(final_metrics, f, indent=2, default=str)
    print(f"\n  Metrics saved: {mpath}")

    print(f"\n{'='*72}")
    print("PHASE 7 COMPLETE — XGBOOST COMPARISON MODEL")
    print(f"{'='*72}")
    print(f"\n  BINARY (Test):")
    b = final_metrics["binary"]
    print(f"    accuracy  : {b['test']['accuracy']:.4f}")
    print(f"    f1_macro  : {b['test']['f1_macro']:.4f}")
    print(f"    roc_auc   : {b['test']['roc_auc']:.4f}")
    print(f"    best_iter : {b['best_iteration']}")
    print(f"\n  MULTICLASS (Test):")
    m = final_metrics["multiclass"]
    print(f"    accuracy  : {m['test']['accuracy']:.4f}")
    print(f"    f1_macro  : {m['test']['f1_macro']:.4f}")
    print(f"    best_iter : {m['best_iteration']}")
    print(f"\n  Elapsed: {elapsed}s")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
# This file was already complete — the patch below is applied via a separate fix script
