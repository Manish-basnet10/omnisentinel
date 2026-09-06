"""
PHASE 2 – Data Cleaning
=======================
Steps: load in day-order → strip col names → drop dup col → normalize labels
       → drop duplicate rows → replace Inf→NaN → median impute → validate → save
"""

import os, sys, json, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

BASE_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
DATA_DIR    = Path("/Users/manishbasnet/Desktop/omnisentinel/SIH DATA SET")
PROC_DIR    = BASE_DIR / "data" / "processed"
METRICS_DIR = BASE_DIR / "results" / "metrics"
PROC_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Chronological file order (Mon→Tue→Wed→Thu-AM→Thu-PM→Fri-AM→Fri-PortScan→Fri-DDoS)
FILE_ORDER = [
    {"order":1,"day":"Monday",   "file":"Monday-WorkingHours.pcap_ISCX.csv",                           "attack_phase":"Benign_Baseline",      "mitre_stage":"None"},
    {"order":2,"day":"Tuesday",  "file":"Tuesday-WorkingHours.pcap_ISCX.csv",                          "attack_phase":"Reconnaissance",       "mitre_stage":"TA0043"},
    {"order":3,"day":"Wednesday","file":"Wednesday-workingHours.pcap_ISCX.csv",                        "attack_phase":"Impact_DoS",            "mitre_stage":"TA0040"},
    {"order":4,"day":"Thursday", "file":"Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",      "attack_phase":"Exploitation_Web",     "mitre_stage":"TA0001"},
    {"order":5,"day":"Thursday", "file":"Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv", "attack_phase":"Lateral_Movement",     "mitre_stage":"TA0008"},
    {"order":6,"day":"Friday",   "file":"Friday-WorkingHours-Morning.pcap_ISCX.csv",                   "attack_phase":"C2_Bot",               "mitre_stage":"TA0011"},
    {"order":7,"day":"Friday",   "file":"Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",        "attack_phase":"Reconnaissance_Scan",  "mitre_stage":"TA0043"},
    {"order":8,"day":"Friday",   "file":"Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",            "attack_phase":"Impact_DDoS",           "mitre_stage":"TA0040"},
]

REMOVED_COLUMNS = [
    {"column":"Fwd Header Length.1","reason":"Exact duplicate of 'Fwd Header Length' (col 34). Zero information gain.","phase":2}
]

def normalize_label(raw):
    raw = str(raw).strip()
    rl = raw.lower()
    if "benign" in rl:    return "BENIGN"
    if "hulk" in rl:      return "DoS_Hulk"
    if "golden" in rl:    return "DoS_GoldenEye"
    if "slowloris" in rl: return "DoS_slowloris"
    if "slowhttp" in rl:  return "DoS_Slowhttptest"
    if "ddos" in rl:      return "DDoS"
    if "portscan" in rl or "port scan" in rl: return "PortScan"
    if "ftp" in rl:       return "FTP-Patator"
    if "ssh" in rl:       return "SSH-Patator"
    if "bot" in rl:       return "Bot"
    if "brute" in rl:     return "WebAttack_BruteForce"
    if "xss" in rl:       return "WebAttack_XSS"
    if "sql" in rl:       return "WebAttack_SQLInjection"
    if "infiltr" in rl:   return "Infiltration"
    if "heartbleed" in rl:return "Heartbleed"
    return raw

def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 2 – DATA CLEANING")
    print("="*72)

    report = {"phase":2,"seed":SEED,"files_processed":[],"steps":[],"removed_columns":REMOVED_COLUMNS}

    # STEP 1: Load
    print("\n[STEP 1] Loading files in chronological order...")
    frames = []
    global_seq = 0
    for meta in FILE_ORDER:
        fpath = DATA_DIR / meta["file"]
        fpath_parquet = DATA_DIR / (meta["file"] + ".parquet")
        
        print(f"  [{meta['order']}] {meta['file']}", end=" ... ", flush=True)
        
        if fpath_parquet.exists():
            df = pd.read_parquet(fpath_parquet)
        elif fpath.exists():
            df = pd.read_csv(fpath, low_memory=False)
        else:
            raise FileNotFoundError(f"Could not find {fpath.name} or {fpath_parquet.name} in {DATA_DIR}")
            
        df.columns = df.columns.str.strip()
        rows_raw = len(df)
        df["day_order"]    = meta["order"]
        df["day_name"]     = meta["day"]
        df["attack_phase"] = meta["attack_phase"]
        df["mitre_stage"]  = meta["mitre_stage"]
        df["seq_idx"]      = range(global_seq, global_seq + len(df))
        global_seq += len(df)
        frames.append(df)
        print(f"{rows_raw:,} rows")
        report["files_processed"].append({"order":meta["order"],"file":meta["file"],"rows_loaded":rows_raw})

    print("  Concatenating...", end=" ", flush=True)
    df = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    rows_init, cols_init = df.shape
    print(f"shape={df.shape}")
    report["pre_cleaning"] = {"rows":rows_init,"cols":cols_init}

    # STEP 2: Remove duplicate column
    print("\n[STEP 2] Removing duplicate column 'Fwd Header Length.1'...")
    if "Fwd Header Length.1" in df.columns and "Fwd Header Length" in df.columns:
        identical = (df["Fwd Header Length"] == df["Fwd Header Length.1"]).all()
        print(f"  Confirmed identical: {identical}")
    df.drop(columns=["Fwd Header Length.1"], inplace=True, errors="ignore")
    print(f"  Shape now: {df.shape}")
    report["steps"].append({"step":2,"action":"drop_dup_col","col":"Fwd Header Length.1"})

    # STEP 3: Normalize labels
    print("\n[STEP 3] Normalizing Label strings...")
    raw_labels = df["Label"].unique().tolist()
    print(f"  Raw labels ({len(raw_labels)}): {raw_labels}")
    df["Label"] = df["Label"].apply(normalize_label)
    norm_labels = sorted(df["Label"].unique().tolist())
    print(f"  Normalized ({len(norm_labels)}): {norm_labels}")
    report["label_normalization"] = {"raw":len(raw_labels),"normalized":len(norm_labels),"labels":norm_labels}
    report["steps"].append({"step":3,"action":"normalize_labels"})

    # STEP 4: Drop duplicate rows
    print("\n[STEP 4] Dropping exact duplicate rows...")
    meta_cols = ["day_order","day_name","attack_phase","mitre_stage","seq_idx"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    rows_before = len(df)
    df.drop_duplicates(subset=feat_cols, keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)
    rows_after = len(df)
    removed = rows_before - rows_after
    print(f"  Before: {rows_before:,}  After: {rows_after:,}  Removed: {removed:,} ({100*removed/rows_before:.2f}%)")
    report["steps"].append({"step":4,"action":"drop_duplicates","removed":removed,"pct":round(100*removed/rows_before,2)})

    # STEP 5: Numeric columns
    non_numeric = meta_cols + ["Label"]
    numeric_cols = [c for c in df.columns if c not in non_numeric]
    print(f"\n[STEP 5] Numeric features: {len(numeric_cols)}")

    # STEP 6: Replace Inf → NaN
    print("\n[STEP 6] Replacing Inf with NaN...")
    inf_count = np.isinf(df[numeric_cols].values).sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    nan_after_inf = int(df[numeric_cols].isnull().sum().sum())
    print(f"  Inf replaced: {inf_count:,}  | Total NaN now: {nan_after_inf:,}")
    report["steps"].append({"step":6,"action":"replace_inf","inf_replaced":int(inf_count),"nan_after":nan_after_inf})

    # STEP 7: Median imputation
    print("\n[STEP 7] Median imputing NaN...")
    cols_with_nan = df[numeric_cols].columns[df[numeric_cols].isnull().any()].tolist()
    print(f"  Columns with NaN: {len(cols_with_nan)} -> {cols_with_nan}")
    medians = {}
    for col in numeric_cols:
        m = float(df[col].median())
        medians[col] = m
        if df[col].isnull().any():
            df[col].fillna(m, inplace=True)
    nan_final = int(df[numeric_cols].isnull().sum().sum())
    print(f"  NaN after imputation: {nan_final}")
    medians_path = PROC_DIR / "phase2_column_medians.json"
    with open(medians_path,"w") as f: json.dump(medians,f,indent=2)
    print(f"  Medians saved: {medians_path}")
    report["steps"].append({"step":7,"action":"median_impute","cols_imputed":len(cols_with_nan),"nan_remaining":nan_final})

    # STEP 8: Final validation
    print("\n[STEP 8] Final validation...")
    assert int(df[numeric_cols].isnull().sum().sum()) == 0, "NaN remain!"
    assert int(np.isinf(df[numeric_cols].values).sum())  == 0, "Inf remain!"
    print("  PASSED: Zero NaN, Zero Inf.")

    # STEP 9: Label distribution
    print("\n[STEP 9] Post-cleaning label distribution:")
    lc = df["Label"].value_counts()
    for lbl, cnt in lc.items():
        print(f"  {lbl:<35} {cnt:>8,}  ({100*cnt/len(df):5.2f}%)")

    # STEP 10: Save
    print("\n[STEP 10] Saving cleaned dataset...")
    out_parquet = PROC_DIR / "cleaned_full.parquet"
    df.to_parquet(out_parquet, index=False, compression="snappy")
    size_mb = os.path.getsize(out_parquet)/(1024*1024)
    print(f"  Saved: {out_parquet}  ({size_mb:.1f} MB)")

    elapsed = round(time.time()-t0, 2)
    report["post_cleaning"] = {
        "rows":len(df),"cols":df.shape[1],
        "nan":0,"inf":0,
        "label_distribution":lc.to_dict()
    }
    report["output_parquet"] = str(out_parquet)
    report["elapsed_seconds"] = elapsed

    rpt_path = METRICS_DIR / "phase2_cleaning_report.json"
    with open(rpt_path,"w") as f: json.dump(report,f,indent=2,default=str)
    print(f"  Report: {rpt_path}")

    print(f"\n{'='*72}")
    print(f"PHASE 2 COMPLETE | Elapsed: {elapsed}s")
    print(f"  Rows:  {rows_init:>10,} → {len(df):>10,}  (removed {rows_init-len(df):,})")
    print(f"  Cols:  {cols_init:>10,} → {df.shape[1]:>10,}")
    print(f"{'='*72}")
    return df, report

if __name__ == "__main__":
    df_cleaned, report = main()
    print("\nFinal columns:")
    for i,c in enumerate(df_cleaned.columns):
        print(f"  [{i:3d}] {c}")
