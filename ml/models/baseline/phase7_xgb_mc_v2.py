"""
Phase 7 — XGBoost Multiclass (v2)
Uses xgb.train() (low-level API) to bypass sklearn label validation.
Maps sparse class IDs → contiguous → back.
"""
import os, json, warnings, time, gc, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)
warnings.filterwarnings("ignore")
SEED = 42; np.random.seed(SEED)

BASE_DIR   = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR   = BASE_DIR/"data"/"processed"
MODELS_DIR = BASE_DIR/"models"
METRICS_DIR= BASE_DIR/"results"/"metrics"
PLOTS_DIR  = BASE_DIR/"results"/"plots"

print("="*72)
print("PHASE 7B – XGBOOST MULTICLASS (LOW-LEVEL API, LABEL REMAP)")
print("="*72)
t0 = time.time()

with open(PROC_DIR/"feature_list.json") as f: feat_def  = json.load(f)
with open(PROC_DIR/"label_encoding.json") as f: label_enc = json.load(f)
FEATURE_COLS = feat_def["all_numeric_features"]
CLASS_NAMES  = label_enc["classes"]

with open(MODELS_DIR/"standard_scaler.pkl","rb") as f: sp = pickle.load(f)
scaler = sp["scaler"]

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

# Remap all labels to contiguous range [0 .. n_train_classes-1]
# We remap based on TRAIN classes only — unseen val/test classes
# will be mapped to their nearest seen class (or kept for evaluation)
train_classes  = sorted(np.unique(y_train_mc).tolist())
n_tc = len(train_classes)
orig2c = {orig: new for new, orig in enumerate(train_classes)}
c2orig = {new: orig for orig, new in orig2c.items()}
print(f"  Train classes: {train_classes}  → mapped to 0..{n_tc-1}")

# Remap train
y_train_c = np.array([orig2c[y] for y in y_train_mc], dtype=np.int32)
# For val/test: map known classes; unknown → 0 (BENIGN) as fallback
y_val_c  = np.array([orig2c.get(y, 0) for y in y_val_mc],  dtype=np.int32)
y_test_c = np.array([orig2c.get(y, 0) for y in y_test_mc], dtype=np.int32)

print(f"  Train contiguous unique: {sorted(np.unique(y_train_c).tolist())}")
print(f"  Val  contiguous unique:  {sorted(np.unique(y_val_c).tolist())}")
print(f"  Test contiguous unique:  {sorted(np.unique(y_test_c).tolist())}")

# DMatrix objects
dtrain = xgb.DMatrix(X_train, label=y_train_c, feature_names=FEATURE_COLS)
dval   = xgb.DMatrix(X_val,   label=y_val_c,   feature_names=FEATURE_COLS)
dtest  = xgb.DMatrix(X_test,  label=y_test_c,  feature_names=FEATURE_COLS)

params = {
    "objective":        "multi:softprob",
    "num_class":        n_tc,
    "max_depth":        6,
    "eta":              0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "tree_method":      "hist",
    "seed":             SEED,
    "nthread":          -1,
}
evals = [(dtrain,"train"),(dval,"val")]
evals_result = {}
print(f"\n  Training (num_class={n_tc}, max_depth=6, eta=0.1)...")
t_fit = time.time()
bst = xgb.train(
    params, dtrain,
    num_boost_round=500,
    evals=evals,
    evals_result=evals_result,
    early_stopping_rounds=20,
    verbose_eval=50
)
fit_time = round(time.time()-t_fit, 2)
best_iter = bst.best_iteration
print(f"\n  Fit time: {fit_time}s  |  Best iteration: {best_iter}")

# Predict — probabilities shape (N, n_tc)
prob_val  = bst.predict(dval).reshape(-1, n_tc)
prob_test = bst.predict(dtest).reshape(-1, n_tc)
pred_val_c  = np.argmax(prob_val,  axis=1)
pred_test_c = np.argmax(prob_test, axis=1)

# Map predictions back to ORIGINAL class space
pred_val_orig  = np.array([c2orig[p] for p in pred_val_c],  dtype=int)
pred_test_orig = np.array([c2orig[p] for p in pred_test_c], dtype=int)

def metrics_mc(y_true, y_pred):
    m = {}
    m["accuracy"]           = float(accuracy_score(y_true, y_pred))
    m["precision_macro"]    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
    m["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    m["recall_macro"]       = float(recall_score(y_true, y_pred,    average="macro",    zero_division=0))
    m["recall_weighted"]    = float(recall_score(y_true, y_pred,    average="weighted", zero_division=0))
    m["f1_macro"]           = float(f1_score(y_true, y_pred,        average="macro",    zero_division=0))
    m["f1_weighted"]        = float(f1_score(y_true, y_pred,        average="weighted", zero_division=0))
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    labels = sorted(set(y_true))
    m["per_class_f1"] = {CLASS_NAMES[i]: round(float(v),4) for i,v in zip(labels, f1_per)}
    return m

met_val  = metrics_mc(y_val_mc,  pred_val_orig)
met_test = metrics_mc(y_test_mc, pred_test_orig)

print(f"\n  Multiclass VAL:")
for k,v in met_val.items():
    if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}")
print(f"\n  Multiclass TEST:")
for k,v in met_test.items():
    if k!="per_class_f1": print(f"    {k:<28}: {v:.4f}")

idx2class = label_enc["idx2class"]
print(f"\n  Per-class F1 (Test):")
for cls, f1v in met_test.get("per_class_f1",{}).items():
    cnt = int(np.sum(y_test_mc == label_enc["class2idx"].get(cls,-1)))
    print(f"    {cls:<35}: F1={f1v:.4f}  (support={cnt})")

y_test_pred_names = [idx2class[str(p)] for p in pred_test_orig]
y_test_true_names = [idx2class[str(t)] for t in y_test_mc]
present = sorted(set(y_test_true_names)|set(y_test_pred_names))
print("\n  Classification Report (Test):")
print(classification_report(y_test_true_names, y_test_pred_names, labels=present, zero_division=0))

# Save booster + remap info
mc_path = MODELS_DIR/"xgb_multiclass.json"  # native XGBoost format
bst.save_model(str(mc_path))
meta = {"orig2c":orig2c,"c2orig":c2orig,"n_tc":n_tc,"train_classes":train_classes,"class_names":CLASS_NAMES}
with open(MODELS_DIR/"xgb_multiclass_meta.json","w") as f: json.dump(meta, f, indent=2)
print(f"\n  Model saved: {mc_path}")

# Confusion matrix
present_int = sorted(set(y_test_mc))
present_names = [CLASS_NAMES[i] for i in present_int]
cm = confusion_matrix(y_test_mc, pred_test_orig, labels=present_int)
fig, ax = plt.subplots(figsize=(8,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=present_names, yticklabels=present_names, ax=ax, linewidths=0.4)
ax.set_title("XGBoost Multiclass — Confusion Matrix (Test)", fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.xticks(rotation=45, ha="right", fontsize=9); plt.tight_layout()
plt.savefig(PLOTS_DIR/"phase7_confusion_matrix_multiclass.png", dpi=150, bbox_inches="tight"); plt.close()

# Training curves
fig, axes = plt.subplots(1,2,figsize=(14,5))
for ai, met in enumerate(["mlogloss","merror"]):
    ax = axes[ai]
    ax.plot(evals_result["train"][met], label="Train", color="#3498db", lw=1.5)
    ax.plot(evals_result["val"][met],   label="Val",   color="#e74c3c", lw=1.5)
    ax.set_xlabel("Boosting Rounds"); ax.set_ylabel(met)
    ax.set_title(f"XGBoost Multiclass — {met}", fontsize=11, fontweight="bold")
    ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(PLOTS_DIR/"phase7_training_curves_multiclass.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"  Plots saved.")

# Feature importance (gain)
fi = bst.get_score(importance_type="gain")
fi_named = {FEATURE_COLS[int(k[1:])]: v for k,v in fi.items() if int(k[1:])<len(FEATURE_COLS)}
fi_sorted = sorted(fi_named.items(), key=lambda x:-x[1])
print(f"\n  Top 15 features (gain — multiclass):")
for feat,g in fi_sorted[:15]:
    print(f"    {feat:<40}: {g:.2f}")

eng_set = set(feat_def["engineered_features"])
top20 = fi_sorted[:20]
colors = ["#e74c3c" if f in eng_set else "#2980b9" for f,_ in top20]
fig, ax = plt.subplots(figsize=(11,7))
ax.barh([x[0] for x in top20[::-1]], [x[1] for x in top20[::-1]], color=colors[::-1], edgecolor="white", height=0.7)
ax.set_xlabel("Gain"); ax.set_title("XGBoost Multiclass — Top 20 Feature Importance (Gain)", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#e74c3c",label="Engineered"),Patch(facecolor="#2980b9",label="Original")], loc="lower right")
plt.tight_layout()
plt.savefig(PLOTS_DIR/"phase7_feature_importance_mc_gain.png", dpi=150, bbox_inches="tight"); plt.close()

# Update phase7_xgb_metrics.json
with open(METRICS_DIR/"phase7_xgb_metrics.json") as f: m7 = json.load(f)
m7["multiclass"]["best_iteration"] = int(best_iter)
m7["multiclass"]["fit_time_s"]     = fit_time
m7["multiclass"]["n_train_classes"]= n_tc
m7["multiclass"]["val"]  = {k:round(v,6) if isinstance(v,float) else v for k,v in met_val.items()}
m7["multiclass"]["test"] = {k:round(v,6) if isinstance(v,float) else v for k,v in met_test.items()}
m7["elapsed_seconds"] = round(time.time()-t0,2)
with open(METRICS_DIR/"phase7_xgb_metrics.json","w") as f: json.dump(m7, f, indent=2, default=str)

# LR vs XGB comparison
with open(METRICS_DIR/"phase6_lr_metrics.json") as f: lr_m = json.load(f)
print(f"\n  {'='*57}")
print(f"  LR vs XGBoost — Test Set Comparison")
print(f"  {'='*57}")
print(f"\n  BINARY:")
print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
print(f"    {'-'*57}")
for metric in ["accuracy","f1_macro","f1_weighted","recall_macro","roc_auc"]:
    lr_v  = lr_m["binary"]["test"].get(metric)
    xgb_v = m7["binary"]["test"].get(metric)
    if lr_v and xgb_v:
        d = xgb_v-lr_v
        print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {'▲' if d>0 else '▼'}{abs(d):>9.4f}")
print(f"\n  MULTICLASS:")
print(f"    {'Metric':<25} {'LR':>10} {'XGB':>10}  {'Delta':>10}")
print(f"    {'-'*57}")
for metric in ["accuracy","f1_macro","f1_weighted","recall_macro"]:
    lr_v  = lr_m["multiclass"]["test"].get(metric)
    xgb_v = met_test.get(metric)
    if lr_v is not None and xgb_v is not None:
        d = xgb_v-lr_v
        print(f"    {metric:<25} {lr_v:>10.4f} {xgb_v:>10.4f}  {'▲' if d>0 else '▼'}{abs(d):>9.4f}")

elapsed = round(time.time()-t0,2)
print(f"\n{'='*72}")
print(f"PHASE 7 COMPLETE | Elapsed: {elapsed}s")
print(f"  BINARY  TEST  acc={m7['binary']['test']['accuracy']:.4f}  f1_macro={m7['binary']['test']['f1_macro']:.4f}  roc_auc={m7['binary']['test']['roc_auc']:.4f}")
print(f"  MC      TEST  acc={met_test['accuracy']:.4f}  f1_macro={met_test['f1_macro']:.4f}")
print(f"{'='*72}")
