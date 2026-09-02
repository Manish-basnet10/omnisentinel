"""
Phase 7 — XGBoost Multiclass Fix
Handles non-contiguous label space (some classes absent from train).
Remaps train labels to 0-indexed contiguous, runs XGBoost,
then maps predictions back to original 15-class space.
"""
import os, json, warnings, time, gc, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score
)
warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)

BASE_DIR   = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR   = BASE_DIR/"data"/"processed"
MODELS_DIR = BASE_DIR/"models"
METRICS_DIR= BASE_DIR/"results"/"metrics"
PLOTS_DIR  = BASE_DIR/"results"/"plots"

def compute_metrics_mc(y_true, y_pred, class_names=None):
    m = {}
    m["accuracy"]           = float(accuracy_score(y_true, y_pred))
    m["precision_macro"]    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    m["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    m["recall_macro"]       = float(recall_score(y_true, y_pred,    average="macro",    zero_division=0))
    m["recall_weighted"]    = float(recall_score(y_true, y_pred,    average="weighted", zero_division=0))
    m["f1_macro"]           = float(f1_score(y_true, y_pred,        average="macro",    zero_division=0))
    m["f1_weighted"]        = float(f1_score(y_true, y_pred,        average="weighted", zero_division=0))
    per_class_f1   = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels_present = sorted(set(y_true))
    if class_names:
        m["per_class_f1"] = {class_names[i]: round(float(v),4)
                              for i,v in zip(labels_present, per_class_f1)}
    return m

t0 = time.time()
print("="*72)
print("PHASE 7B – XGBOOST MULTICLASS (FIXED LABEL REMAPPING)")
print("="*72)

with open(PROC_DIR/"feature_list.json") as f: feat_def = json.load(f)
with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)
FEATURE_COLS = feat_def["all_numeric_features"]
CLASS_NAMES  = label_enc["classes"]

with open(MODELS_DIR/"standard_scaler.pkl","rb") as f: scaler_pkg = pickle.load(f)
scaler = scaler_pkg["scaler"]

df_train = pd.read_parquet(PROC_DIR/"state_train.parquet")
df_val   = pd.read_parquet(PROC_DIR/"state_val.parquet")
df_test  = pd.read_parquet(PROC_DIR/"state_test.parquet")

X_train = scaler.transform(df_train[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)
X_val   = scaler.transform(df_val[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)
X_test  = scaler.transform(df_test[FEATURE_COLS].values.astype(np.float32)).astype(np.float32)

y_train_mc = df_train["label_multiclass"].values.astype(int)
y_val_mc   = df_val["label_multiclass"].values.astype(int)
y_test_mc  = df_test["label_multiclass"].values.astype(int)
del df_train, df_val, df_test; gc.collect()

# ── Label remapping (contiguous for XGBoost) ──────────────────────────────
# Only remap TRAIN labels; val and test may have different sets
train_classes = sorted(np.unique(y_train_mc).tolist())
# Extend with val+test classes so inverse map is complete
all_eval_classes = sorted(set(np.unique(y_val_mc).tolist()) | set(np.unique(y_test_mc).tolist()))
# Build global remap over all classes seen in any split
all_classes_seen = sorted(set(train_classes) | set(all_eval_classes))

orig2remap = {orig: new for new, orig in enumerate(all_classes_seen)}
remap2orig = {new: orig for orig, new in orig2remap.items()}

y_train_remap = np.array([orig2remap[y] for y in y_train_mc], dtype=int)
y_val_remap   = np.array([orig2remap.get(y, -1) for y in y_val_mc],  dtype=int)
y_test_remap  = np.array([orig2remap.get(y, -1) for y in y_test_mc], dtype=int)

n_remap = len(all_classes_seen)
print(f"\n  Original class indices (all seen): {all_classes_seen}")
print(f"  Remapped to 0..{n_remap-1}")
print(f"  Train unique (remapped): {sorted(np.unique(y_train_remap).tolist())}")
print(f"  Val  unique (remapped): {sorted(np.unique(y_val_remap).tolist())}")
print(f"  Test unique (remapped): {sorted(np.unique(y_test_remap).tolist())}")

xgb_params_mc = dict(
    objective="multi:softprob", num_class=n_remap,
    n_estimators=500, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    tree_method="hist", eval_metric=["mlogloss","merror"],
    early_stopping_rounds=20, random_state=SEED, n_jobs=-1, verbosity=0
)
print(f"\n  Training XGBoost Multiclass (num_class={n_remap})...")
xgb_mc = xgb.XGBClassifier(**xgb_params_mc)
t_fit = time.time()
xgb_mc.fit(
    X_train, y_train_remap,
    eval_set=[(X_train, y_train_remap), (X_val, y_val_remap)],
    verbose=50
)
fit_time_mc = round(time.time()-t_fit, 2)
best_iter = xgb_mc.best_iteration
print(f"\n  Fit time: {fit_time_mc}s  |  Best iteration: {best_iter}")

# Predict and remap back to original class space
y_val_pred_remap  = xgb_mc.predict(X_val)
y_val_pred_mc     = np.array([remap2orig[p] for p in y_val_pred_remap], dtype=int)
y_test_pred_remap = xgb_mc.predict(X_test)
y_test_pred_mc    = np.array([remap2orig[p] for p in y_test_pred_remap], dtype=int)

idx2class = label_enc["idx2class"]
metrics_val_mc  = compute_metrics_mc(y_val_mc, y_val_pred_mc,  class_names=CLASS_NAMES)
metrics_test_mc = compute_metrics_mc(y_test_mc, y_test_pred_mc, class_names=CLASS_NAMES)

print(f"\n  Multiclass VAL:")
for k,v in metrics_val_mc.items():
    if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")

print(f"\n  Multiclass TEST:")
for k,v in metrics_test_mc.items():
    if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}" if isinstance(v,float) else f"    {k}: {v}")

print(f"\n  Per-class F1 (Test):")
for cls, f1v in metrics_test_mc.get("per_class_f1",{}).items():
    orig_idx = label_enc["class2idx"].get(cls, -1)
    cnt = int(np.sum(y_test_mc == orig_idx))
    print(f"    {cls:<35}: F1={f1v:.4f}  (support={cnt})")

y_test_pred_names = [idx2class[str(p)] for p in y_test_pred_mc]
y_test_true_names = [idx2class[str(t)] for t in y_test_mc]
present_classes   = sorted(set(y_test_true_names)|set(y_test_pred_names))
print("\n  Classification Report (Test — Multiclass):")
print(classification_report(y_test_true_names, y_test_pred_names,
                              labels=present_classes, zero_division=0))

# Save model + remap info
xgb_mc_path = MODELS_DIR/"xgb_multiclass.pkl"
with open(xgb_mc_path,"wb") as f:
    pickle.dump({"model":xgb_mc,"orig2remap":orig2remap,"remap2orig":remap2orig,
                 "n_remap":n_remap,"class_names":CLASS_NAMES}, f)
print(f"\n  Model saved: {xgb_mc_path}")

# Confusion matrix (test)
present_label_ints = sorted(set(y_test_mc))
present_names      = [CLASS_NAMES[i] for i in present_label_ints]
cm = confusion_matrix(y_test_mc, y_test_pred_mc, labels=present_label_ints)
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names,
            ax=ax, linewidths=0.4)
ax.set_title("XGBoost Multiclass — Confusion Matrix (Test)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.tight_layout()
cm_path = PLOTS_DIR/"phase7_confusion_matrix_multiclass.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight"); plt.close()
print(f"  Confusion matrix saved: {cm_path.name}")

# Training curves
evals_mc = xgb_mc.evals_result()
fig, axes = plt.subplots(1, 2, figsize=(14,5))
for ai, metric in enumerate(["mlogloss","merror"]):
    ax = axes[ai]
    ax.plot(evals_mc["validation_0"][metric], label="Train", color="#3498db", lw=1.5)
    ax.plot(evals_mc["validation_1"][metric], label="Val",   color="#e74c3c", lw=1.5)
    ax.set_xlabel("Boosting Rounds"); ax.set_ylabel(metric)
    ax.set_title(f"XGBoost Multiclass — {metric}", fontsize=11, fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"phase7_training_curves_multiclass.png", dpi=150, bbox_inches="tight"); plt.close()

# Load Phase 6 LR metrics and append XGBoost multiclass
with open(METRICS_DIR/"phase7_xgb_metrics.json") as f: m7 = json.load(f)
m7["multiclass"]["best_iteration"] = int(best_iter)
m7["multiclass"]["fit_time_s"]     = fit_time_mc
m7["multiclass"]["val"]  = {k:round(v,6) if isinstance(v,float) else v for k,v in metrics_val_mc.items()}
m7["multiclass"]["test"] = {k:round(v,6) if isinstance(v,float) else v for k,v in metrics_test_mc.items()}
m7["multiclass"]["label_remap"] = {"orig2remap":orig2remap,"n_remap":n_remap}
m7["elapsed_seconds"] = round(time.time()-t0,2)
with open(METRICS_DIR/"phase7_xgb_metrics.json","w") as f:
    json.dump(m7, f, indent=2, default=str)

# Compare with LR
print("\n  LR vs XGBoost comparison (Test):")
with open(METRICS_DIR/"phase6_lr_metrics.json") as f: lr_m = json.load(f)
print(f"\n  BINARY:")
print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
print(f"    {'-'*57}")
for metric in ["accuracy","f1_macro","recall_macro","roc_auc"]:
    lr_v  = lr_m["binary"]["test"].get(metric)
    xgb_v = m7["binary"]["test"].get(metric)
    if lr_v and xgb_v:
        delta = xgb_v - lr_v
        print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {'▲' if delta>0 else '▼'}{abs(delta):>9.4f}")
print(f"\n  MULTICLASS:")
print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
print(f"    {'-'*57}")
for metric in ["accuracy","f1_macro","recall_macro"]:
    lr_v  = lr_m["multiclass"]["test"].get(metric)
    xgb_v = metrics_test_mc.get(metric)
    if lr_v is not None and xgb_v is not None:
        delta = xgb_v - lr_v
        print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {'▲' if delta>0 else '▼'}{abs(delta):>9.4f}")

elapsed = round(time.time()-t0,2)
print(f"\n{'='*72}")
print(f"PHASE 7 MULTICLASS COMPLETE | Elapsed: {elapsed}s")
print(f"  Accuracy  (test): {metrics_test_mc['accuracy']:.4f}")
print(f"  F1-macro  (test): {metrics_test_mc['f1_macro']:.4f}")
print(f"  Best iter       : {best_iter}")
print(f"{'='*72}")
