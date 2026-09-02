"""
PHASE 3 – Feature Engineering (fixed order)
============================================
Order: Load → NZV drop → Engineer NEW features (using all raw cols) 
       → Correlation drop → Encode labels → Save
"""

import os, json, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR    = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PROC_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

CORR_THRESHOLD   = 0.97
NZV_STD_THRESHOLD = 0.0

FEATURE_INVENTORY = {
    "available_groups": {
        "flow_volume":          ["Flow Duration","Total Fwd Packets","Total Backward Packets","Total Length of Fwd Packets","Total Length of Bwd Packets"],
        "packet_length_stats":  ["Fwd Packet Length Max","Fwd Packet Length Min","Fwd Packet Length Mean","Fwd Packet Length Std","Bwd Packet Length Max","Bwd Packet Length Min","Bwd Packet Length Mean","Bwd Packet Length Std","Min Packet Length","Max Packet Length","Packet Length Mean","Packet Length Std","Packet Length Variance","Average Packet Size","Avg Fwd Segment Size","Avg Bwd Segment Size"],
        "inter_arrival_time":   ["Flow IAT Mean","Flow IAT Std","Flow IAT Max","Flow IAT Min","Fwd IAT Total","Fwd IAT Mean","Fwd IAT Std","Fwd IAT Max","Fwd IAT Min","Bwd IAT Total","Bwd IAT Mean","Bwd IAT Std","Bwd IAT Max","Bwd IAT Min"],
        "tcp_flags":            ["Fwd PSH Flags","Bwd PSH Flags","Fwd URG Flags","Bwd URG Flags","FIN Flag Count","SYN Flag Count","RST Flag Count","PSH Flag Count","ACK Flag Count","URG Flag Count","CWE Flag Count","ECE Flag Count"],
        "rates":                ["Flow Bytes/s","Flow Packets/s","Fwd Packets/s","Bwd Packets/s"],
        "header":               ["Fwd Header Length","Bwd Header Length"],
        "bulk_stats":           ["Fwd Avg Bytes/Bulk","Fwd Avg Packets/Bulk","Fwd Avg Bulk Rate","Bwd Avg Bytes/Bulk","Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate"],
        "subflow":              ["Subflow Fwd Packets","Subflow Fwd Bytes","Subflow Bwd Packets","Subflow Bwd Bytes"],
        "tcp_window_segment":   ["Init_Win_bytes_forward","Init_Win_bytes_backward","act_data_pkt_fwd","min_seg_size_forward"],
        "active_idle":          ["Active Mean","Active Std","Active Max","Active Min","Idle Mean","Idle Std","Idle Max","Idle Min"],
        "network_id":           ["Destination Port","Down/Up Ratio"]
    },
    "missing_with_reason": {
        "Timestamp_wallclock":      "Stripped by CICFlowMeter in MachineLearningCSV variant",
        "Source_IP":                "Stripped by CICFlowMeter",
        "Destination_IP":           "Stripped by CICFlowMeter",
        "Source_Port":              "Stripped by CICFlowMeter",
        "IP_TTL":                   "PCAP-level only — not in CICFlowMeter",
        "IP_Fragmentation_flags":   "PCAP-level only",
        "TCP_Sequence_Ack_numbers": "PCAP-level only",
        "TCP_Retransmission_count": "Not exported by CICFlowMeter",
        "TCP_OutOfOrder_packets":   "Not exported by CICFlowMeter",
        "Payload_DPI_features":     "Raw PCAP required",
        "ICMP_type_code":           "Not in CICFlowMeter output",
        "DNS_features":             "Not in CICFlowMeter output",
        "HTTP_HTTPS_headers":       "App-layer — not extracted",
        "Per_packet_IAT":           "Only flow-aggregate IAT available"
    }
}

def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 3 – FEATURE ENGINEERING")
    print("="*72)

    report = {
        "phase": 3, "seed": SEED,
        "feature_inventory": FEATURE_INVENTORY,
        "removed_features": [], "steps": []
    }

    # ── STEP 1: Load ───────────────────────────────────────────────────────
    print("\n[STEP 1] Loading cleaned_full.parquet...")
    df = pd.read_parquet(PROC_DIR / "cleaned_full.parquet")
    print(f"  Shape: {df.shape}")

    META_COLS    = ["day_order","day_name","attack_phase","mitre_stage","seq_idx","Label"]
    NUMERIC_COLS = [c for c in df.columns if c not in META_COLS]
    print(f"  Numeric features: {len(NUMERIC_COLS)}")

    # ── STEP 2: Feature inventory print ───────────────────────────────────
    print("\n[STEP 2] Feature Inventory")
    total_avail = sum(len(v) for v in FEATURE_INVENTORY["available_groups"].values())
    print(f"  Available features in raw dataset: {total_avail} across {len(FEATURE_INVENTORY['available_groups'])} groups")
    print(f"  Missing features (documented): {len(FEATURE_INVENTORY['missing_with_reason'])}")
    for k,v in FEATURE_INVENTORY["missing_with_reason"].items():
        print(f"    MISSING: {k}  — {v}")

    # ── STEP 3: Remove near-zero-variance features (std == 0) ──────────────
    print("\n[STEP 3] Removing zero-variance features...")
    stds = df[NUMERIC_COLS].std()
    nzv_cols = stds[stds <= NZV_STD_THRESHOLD].index.tolist()
    print(f"  Zero-variance cols found: {len(nzv_cols)}")
    for col in nzv_cols:
        uniq = df[col].unique()[:3].tolist()
        print(f"    REMOVED: {col}  (constant: {uniq})")
        report["removed_features"].append({
            "column": col, "phase": 3,
            "reason": f"Zero variance (std=0). Constant value: {uniq}. No information for any model."
        })
    df.drop(columns=nzv_cols, inplace=True)
    NUMERIC_COLS = [c for c in df.columns if c not in META_COLS]
    print(f"  Shape: {df.shape}  |  Remaining numeric: {len(NUMERIC_COLS)}")
    report["steps"].append({"step":3,"removed_nzv":nzv_cols})

    # ── STEP 4: Engineer NEW features (BEFORE correlation drop) ────────────
    # Must do this FIRST so we can use all raw columns in formulas
    print("\n[STEP 4] Engineering new features (before corr drop)...")

    # Safe column getter: returns zeros if col was dropped in NZV step
    def safe_col(col):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)

    # Ratio features
    df["pkt_fwd_bwd_ratio"]    = safe_col("Total Fwd Packets") / (safe_col("Total Backward Packets") + 1)
    df["byte_fwd_bwd_ratio"]   = safe_col("Total Length of Fwd Packets") / (safe_col("Total Length of Bwd Packets") + 1)
    df["avg_pkt_size_ratio"]   = safe_col("Fwd Packet Length Mean") / (safe_col("Bwd Packet Length Mean") + 1)
    df["fwd_bwd_iat_ratio"]    = safe_col("Fwd IAT Mean") / (safe_col("Bwd IAT Mean") + 1)
    df["active_idle_ratio"]    = safe_col("Active Mean") / (safe_col("Idle Mean") + 1)
    df["syn_fin_ratio"]        = safe_col("SYN Flag Count") / (safe_col("FIN Flag Count") + 1)
    df["window_size_ratio"]    = safe_col("Init_Win_bytes_forward") / (safe_col("Init_Win_bytes_backward") + 1)
    df["total_packets"]        = safe_col("Total Fwd Packets") + safe_col("Total Backward Packets")
    df["total_bytes"]          = safe_col("Total Length of Fwd Packets") + safe_col("Total Length of Bwd Packets")

    # Flag density
    flag_cols = [c for c in ["FIN Flag Count","SYN Flag Count","RST Flag Count",
                               "PSH Flag Count","ACK Flag Count","URG Flag Count"] if c in df.columns]
    df["flag_density_per_pkt"] = df[flag_cols].sum(axis=1) / (df["total_packets"] + 1)

    # Log1p transforms
    log_map = {
        "log1p_flow_duration": "Flow Duration",
        "log1p_fwd_pkts":      "Total Fwd Packets",
        "log1p_bwd_pkts":      "Total Backward Packets",
        "log1p_flow_bytes_s":  "Flow Bytes/s",
        "log1p_flow_pkts_s":   "Flow Packets/s",
        "log1p_pkt_len_mean":  "Packet Length Mean",
        "log1p_idle_mean":     "Idle Mean",
        "log1p_active_mean":   "Active Mean",
    }
    applied_logs = []
    for new_col, src_col in log_map.items():
        if src_col in df.columns:
            df[new_col] = np.log1p(df[src_col].clip(lower=0))
            applied_logs.append(new_col)

    NEW_FEATURES = ["pkt_fwd_bwd_ratio","byte_fwd_bwd_ratio","avg_pkt_size_ratio",
                    "fwd_bwd_iat_ratio","active_idle_ratio","syn_fin_ratio",
                    "window_size_ratio","total_packets","total_bytes",
                    "flag_density_per_pkt"] + applied_logs

    print(f"  Engineered features added ({len(NEW_FEATURES)}):")
    for f in NEW_FEATURES:
        print(f"    + {f}")

    # Fix any NaN/Inf in engineered features immediately
    for col in NEW_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

    report["steps"].append({"step":4,"engineered_features":NEW_FEATURES})

    # ── STEP 5: Remove highly-correlated features ──────────────────────────
    NUMERIC_COLS = [c for c in df.columns if c not in META_COLS + ["label_binary","label_multiclass"]]
    print(f"\n[STEP 5] Correlation analysis (|r|>{CORR_THRESHOLD}, n_sample=50k)...")
    sample_df = df[NUMERIC_COLS].sample(min(50000, len(df)), random_state=SEED)
    corr_matrix = sample_df.corr(method="pearson").abs()
    print("  Correlation matrix computed.")

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr, corr_pairs = [], []
    for col in upper.columns:
        high_corr = upper[col][upper[col] > CORR_THRESHOLD]
        for partner, r_val in high_corr.items():
            if col not in to_drop_corr:
                to_drop_corr.append(col)
                corr_pairs.append({"dropped": col, "kept": partner, "pearson_r": round(float(r_val), 4)})

    print(f"  Features dropped (|r|>{CORR_THRESHOLD}): {len(to_drop_corr)}")
    for p in corr_pairs:
        print(f"    DROP '{p['dropped']}'  r={p['pearson_r']} with '{p['kept']}'")
        report["removed_features"].append({
            "column": p["dropped"], "phase": 3,
            "reason": f"Pearson |r|={p['pearson_r']} > {CORR_THRESHOLD} with '{p['kept']}'. Redundant.",
            "kept_partner": p["kept"]
        })

    df.drop(columns=to_drop_corr, inplace=True)
    NUMERIC_COLS = [c for c in df.columns if c not in META_COLS + ["label_binary","label_multiclass"]]
    print(f"  Shape after corr drop: {df.shape}  |  Remaining numeric: {len(NUMERIC_COLS)}")
    report["steps"].append({"step":5,"removed_corr":to_drop_corr,"pairs":corr_pairs})

    # ── STEP 6: Encode labels ─────────────────────────────────────────────
    print("\n[STEP 6] Encoding labels...")
    df["label_binary"] = (df["Label"] != "BENIGN").astype(int)
    classes = sorted(df["Label"].unique().tolist())
    class2idx = {c: i for i, c in enumerate(classes)}
    idx2class  = {str(i): c for c, i in class2idx.items()}
    df["label_multiclass"] = df["Label"].map(class2idx)
    print(f"  Binary label: 0=BENIGN, 1=ATTACK")
    print(f"  Multi-class ({len(classes)} classes):")
    for c, i in class2idx.items():
        cnt = (df["Label"] == c).sum()
        print(f"    {i:2d}  {c:<35}  {cnt:>8,}")
    class_path = PROC_DIR / "label_encoding.json"
    with open(class_path, "w") as f:
        json.dump({"class2idx":class2idx,"idx2class":idx2class,"classes":classes}, f, indent=2)
    print(f"  Saved: {class_path}")
    report["steps"].append({"step":6,"class2idx":class2idx})

    # ── STEP 7: Final feature list ────────────────────────────────────────
    print("\n[STEP 7] Final feature list...")
    FINAL_NUMERIC = [c for c in df.columns if c not in META_COLS + ["label_binary","label_multiclass"]]
    ORIGINAL_KEPT = [f for f in FINAL_NUMERIC if f not in NEW_FEATURES]
    ENG_KEPT      = [f for f in FINAL_NUMERIC if f in NEW_FEATURES]
    print(f"  Total numeric features: {len(FINAL_NUMERIC)}")
    print(f"    Original (kept after selection): {len(ORIGINAL_KEPT)}")
    print(f"    Engineered (new):                {len(ENG_KEPT)}")
    print(f"  Full feature list:")
    for i, f in enumerate(FINAL_NUMERIC):
        tag = " [ENG]" if f in NEW_FEATURES else ""
        print(f"    [{i:3d}] {f}{tag}")

    feat_list = {
        "all_numeric_features":   FINAL_NUMERIC,
        "original_features":      ORIGINAL_KEPT,
        "engineered_features":    ENG_KEPT,
        "meta_columns":           META_COLS,
        "label_columns":          ["Label","label_binary","label_multiclass"],
        "total_feature_count":    len(FINAL_NUMERIC)
    }
    feat_path = PROC_DIR / "feature_list.json"
    with open(feat_path, "w") as f:
        json.dump(feat_list, f, indent=2)
    print(f"  Saved: {feat_path}")

    # ── STEP 8: Final validation ──────────────────────────────────────────
    print("\n[STEP 8] Validation...")
    final_nan = int(df[FINAL_NUMERIC].isnull().sum().sum())
    final_inf = int(np.isinf(df[FINAL_NUMERIC].select_dtypes(include=[np.number]).values).sum())
    print(f"  NaN: {final_nan}  |  Inf: {final_inf}")
    assert final_nan == 0 and final_inf == 0, "NaN/Inf remain after engineering!"
    print("  PASSED: Zero NaN, Zero Inf.")

    # ── STEP 9: Save ─────────────────────────────────────────────────────
    print("\n[STEP 9] Saving engineered dataset...")
    out_path = PROC_DIR / "features_engineered.parquet"
    df.to_parquet(out_path, index=False, compression="snappy")
    size_mb = os.path.getsize(out_path) / (1024*1024)
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")

    # ── STEP 10: Summary stats ────────────────────────────────────────────
    print(f"\n[STEP 10] Summary stats for engineered features:")
    stats = df[ENG_KEPT].describe().T[["mean","std","min","max"]]
    print(stats.round(4).to_string())

    # ── Save removed registry ─────────────────────────────────────────────
    reg_path = PROC_DIR / "removed_features_registry.json"
    with open(reg_path, "w") as f:
        json.dump(report["removed_features"], f, indent=2)
    print(f"\n  Removed features registry: {reg_path}")

    # ── Finalize report ───────────────────────────────────────────────────
    elapsed = round(time.time()-t0, 2)
    report["final_shape"] = {"rows": len(df), "cols": df.shape[1]}
    report["final_numeric_count"] = len(FINAL_NUMERIC)
    report["original_features_kept"] = ORIGINAL_KEPT
    report["engineered_features_added"] = ENG_KEPT
    report["elapsed_seconds"] = elapsed
    rpt_path = METRICS_DIR / "phase3_feature_report.json"
    with open(rpt_path,"w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Phase 3 report: {rpt_path}")

    print(f"\n{'='*72}")
    print(f"PHASE 3 COMPLETE | Elapsed: {elapsed}s")
    print(f"  Dataset: {df.shape}")
    print(f"  Total numeric features: {len(FINAL_NUMERIC)}")
    print(f"    - Zero-variance removed:    {len(nzv_cols)}")
    print(f"    - High-corr removed:        {len(to_drop_corr)}")
    print(f"    - Original features kept:   {len(ORIGINAL_KEPT)}")
    print(f"    - New engineered features:  {len(ENG_KEPT)}")
    print(f"{'='*72}")
    return df, report

if __name__ == "__main__":
    df_feat, report = main()
