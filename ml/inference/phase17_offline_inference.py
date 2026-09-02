"""
PHASE 17 – Offline Inference Engine (CSV → JSON)
=================================================
Standalone batch inference: takes a CSV of raw network flows,
runs the GRU World Model K-step rollout, and writes a JSON report.

Usage:
  python phase17_offline_inference.py --input flows.csv --output predictions.json
  python phase17_offline_inference.py --demo   # uses test parquet as demo input
"""
import json, pickle, argparse, warnings, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

BASE_DIR   = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
WM_DIR     = BASE_DIR / "models" / "world_model"

SEQ_LEN      = 10
ROLLOUT_K    = 8
N_CLASSES_MC = 15
RISK_DISCOUNT = 0.85

MITRE_MAP = {
    0:  {"label":"BENIGN",               "tactic":None,                "stage":0,"severity":0},
    1:  {"label":"Bot",                  "tactic":"Command & Control",  "stage":6,"severity":5},
    2:  {"label":"DDoS",                 "tactic":"Impact",             "stage":7,"severity":5},
    3:  {"label":"DoS_GoldenEye",        "tactic":"Impact",             "stage":7,"severity":4},
    4:  {"label":"DoS_Hulk",             "tactic":"Impact",             "stage":7,"severity":4},
    5:  {"label":"DoS_Slowhttptest",     "tactic":"Impact",             "stage":7,"severity":3},
    6:  {"label":"DoS_slowloris",        "tactic":"Impact",             "stage":7,"severity":3},
    7:  {"label":"FTP-Patator",          "tactic":"Credential Access",  "stage":3,"severity":3},
    8:  {"label":"Heartbleed",           "tactic":"Initial Access",     "stage":2,"severity":5},
    9:  {"label":"Infiltration",         "tactic":"Lateral Movement",   "stage":5,"severity":5},
    10: {"label":"PortScan",             "tactic":"Discovery",          "stage":1,"severity":2},
    11: {"label":"SSH-Patator",          "tactic":"Credential Access",  "stage":3,"severity":3},
    12: {"label":"WebAttack_BruteForce", "tactic":"Credential Access",  "stage":3,"severity":3},
    13: {"label":"WebAttack_SQL_Inj",    "tactic":"Initial Access",     "stage":2,"severity":5},
    14: {"label":"WebAttack_XSS",        "tactic":"Execution",          "stage":4,"severity":3},
}
SMAP = {i: MITRE_MAP[i]["severity"] for i in range(N_CLASSES_MC)}


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


def load_model():
    with open(PROC_DIR / "feature_list.json")    as f: feat_def = json.load(f)
    with open(MODELS_DIR / "standard_scaler.pkl","rb") as f: scaler_pkg = pickle.load(f)
    scaler  = scaler_pkg["scaler"]
    feats   = feat_def["all_numeric_features"]
    ckpt    = torch.load(WM_DIR / "gru_world_model.pt", map_location="cpu")
    hp      = ckpt["hyperparams"]
    model   = GRUWorldModel(hp["input_dim"], hp["hidden_dim"], hp["num_layers"],
                             hp["n_classes_mc"], hp["dropout"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, scaler, feats


def rollout(model, seq_scaled, K=ROLLOUT_K):
    window = seq_scaled.copy()
    ap, cp = [], []
    with torch.no_grad():
        for _ in range(K):
            x = torch.tensor(window[np.newaxis], dtype=torch.float32)
            ps, pb, pm = model(x)
            ap.append(float(F.softmax(pb, dim=1)[0, 1]))
            cp.append(F.softmax(pm, dim=1)[0].numpy().tolist())
            window = np.vstack([window[1:], ps[0].numpy()[np.newaxis]])
    return np.array(ap), np.array(cp)


def risk_score(ap, cp):
    K    = len(ap)
    dmax = sum(RISK_DISCOUNT**k for k in range(K)) * 5.0
    raw  = sum(RISK_DISCOUNT**k * ap[k] * (1 + SMAP.get(int(np.argmax(cp[k])), 0)/5.0) for k in range(K))
    return round(min(raw / (dmax * 2.0/5.0) * 100, 100), 2)


def risk_level(score):
    if score >= 85: return "CRITICAL"
    if score >= 65: return "HIGH"
    if score >= 40: return "MEDIUM"
    if score >= 20: return "LOW"
    return "SAFE"


def run_inference(df, model, scaler, feats, K=ROLLOUT_K):
    """
    df: DataFrame with columns matching feats (raw, unscaled).
    Returns list of prediction dicts, one per valid sequence window.
    """
    X_raw    = df[feats].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    n        = len(X_raw)
    results  = []
    t0       = time.time()

    for start in range(0, n - SEQ_LEN):
        end      = start + SEQ_LEN
        seq      = X_scaled[start:end]
        ap, cp   = rollout(model, seq, K=K)
        rs       = risk_score(ap, cp)
        rl       = risk_level(rs)
        top_cls  = int(np.argmax(cp[0]))
        mitre    = MITRE_MAP.get(top_cls, MITRE_MAP[0])
        results.append({
            "window_start_row": int(start),
            "window_end_row":   int(end - 1),
            "risk_score":       rs,
            "risk_level":       rl,
            "attack_prob_timeline": [round(float(p), 4) for p in ap],
            "predicted_class_k1":  mitre["label"],
            "predicted_tactic_k1": mitre["tactic"],
            "mitre_stage_k1":      mitre["stage"],
            "severity_k1":         mitre["severity"],
        })

        if (start + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {start+1}/{n-SEQ_LEN} windows  ({elapsed:.1f}s elapsed)")

    return results


def main():
    parser = argparse.ArgumentParser(description="OmniSentinel Offline Inference Engine")
    parser.add_argument("--input",  type=str, default=None, help="Input CSV file path")
    parser.add_argument("--output", type=str, default="predictions.json", help="Output JSON file path")
    parser.add_argument("--demo",   action="store_true", help="Run demo mode on test parquet (first 200 rows)")
    parser.add_argument("--rollout-k", type=int, default=ROLLOUT_K, help="Number of rollout steps")
    args = parser.parse_args()

    print("=" * 70)
    print("OMNISENTINEL – OFFLINE INFERENCE ENGINE (CSV → JSON)")
    print("=" * 70)

    print("\n[1] Loading model and scaler...")
    model, scaler, feats = load_model()
    print(f"    GRU World Model loaded | Features: {len(feats)} | Rollout K={args.rollout_k}")

    if args.demo:
        print("\n[2] DEMO MODE — loading first 500 rows from state_test.parquet...")
        df = pd.read_parquet(PROC_DIR / "state_test.parquet").head(500)
        # Check all feature columns are present
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"    WARNING: {len(missing)} features missing — filling with 0")
            for f in missing: df[f] = 0.0
        out_path = str(BASE_DIR / "results" / "predictions" / "demo_predictions.json")
    elif args.input:
        print(f"\n[2] Loading input CSV: {args.input}")
        df = pd.read_csv(args.input)
        missing = [f for f in feats if f not in df.columns]
        if missing:
            print(f"    WARNING: {len(missing)} features missing — filling with 0")
            for f in missing: df[f] = 0.0
        out_path = args.output
    else:
        print("ERROR: provide --input <csv> or --demo")
        return

    print(f"\n[3] Running inference on {len(df)} rows...")
    t0      = time.time()
    results = run_inference(df, model, scaler, feats, K=args.rollout_k)
    elapsed = time.time() - t0

    # Summary
    n_crit = sum(1 for r in results if r["risk_level"] == "CRITICAL")
    n_high = sum(1 for r in results if r["risk_level"] == "HIGH")

    report = {
        "model":          "GRU World Model (Phase 8)",
        "rollout_k":      args.rollout_k,
        "n_input_rows":   int(len(df)),
        "n_windows":      int(len(results)),
        "seq_len":        SEQ_LEN,
        "inference_time_s": round(elapsed, 2),
        "ms_per_window":  round(elapsed / max(len(results), 1) * 1000, 3),
        "alerts": {
            "CRITICAL": int(n_crit),
            "HIGH":     int(n_high),
            "MEDIUM":   int(sum(1 for r in results if r["risk_level"] == "MEDIUM")),
            "LOW":      int(sum(1 for r in results if r["risk_level"] == "LOW")),
            "SAFE":     int(sum(1 for r in results if r["risk_level"] == "SAFE")),
        },
        "predictions": results
    }

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print(f"  Windows processed: {len(results):,}")
    print(f"  Total time:        {elapsed:.2f}s  ({report['ms_per_window']}ms/window)")
    print(f"  CRITICAL alerts:   {n_crit}")
    print(f"  HIGH alerts:       {n_high}")
    print(f"  Output saved to:   {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
