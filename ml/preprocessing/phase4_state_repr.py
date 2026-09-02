"""
PHASE 4 – Network State Representation
========================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

DESIGN
------
A "Network State" S_t is a fixed-dimension vector that captures the
observable properties of ONE network flow at temporal position t.

Since the MachineLearningCSV variant has NO wall-clock timestamps and NO
IP addresses, S_t is defined at the FLOW level, ordered by:
  (1) day_order  — file / day number (chronological week ordering)
  (2) seq_idx    — row position within each day file (capture order proxy)

State vector S_t contains the 60 engineered numeric features.
State transition target S_{t+1} is the next row in sequence order.

The label at t (label_binary, label_multiclass) is included as a state
component so the World Model can learn attack-class transitions.

TRAIN / VAL / TEST SPLIT  (Chronological — NO shuffle)
------------------------------------------------------
  Train : day_order 1–5  (Mon, Tue, Wed, Thu-AM, Thu-PM)  ~attack build-up
  Val   : day_order 6    (Fri-AM: Bot)                     ~mid-attack
  Test  : day_order 7–8  (Fri PortScan + Fri DDoS)         ~peak attack

Rationale: Test set contains DDoS + PortScan — two distinct attack types
the model has NOT trained on in their Friday context. This tests temporal
generalization across the attack kill chain.

Outputs
-------
  data/processed/state_train.parquet
  data/processed/state_val.parquet
  data/processed/state_test.parquet
  data/processed/state_definition.json
  results/metrics/phase4_state_report.json
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

# Chronological split boundaries
TRAIN_DAYS = [1, 2, 3, 4, 5]    # Mon – Thu-PM
VAL_DAYS   = [6]                  # Fri-AM (Bot)
TEST_DAYS  = [7, 8]               # Fri PortScan + Fri DDoS

def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 4 – NETWORK STATE REPRESENTATION")
    print("="*72)

    report = {"phase": 4, "seed": SEED, "steps": []}

    # ── STEP 1: Load engineered dataset ───────────────────────────────────
    print("\n[STEP 1] Loading features_engineered.parquet...")
    df = pd.read_parquet(PROC_DIR / "features_engineered.parquet")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # ── STEP 2: Load feature list ─────────────────────────────────────────
    print("\n[STEP 2] Loading feature definitions...")
    with open(PROC_DIR / "feature_list.json") as f:
        feat_def = json.load(f)
    with open(PROC_DIR / "label_encoding.json") as f:
        label_enc = json.load(f)

    NUMERIC_FEATURES  = feat_def["all_numeric_features"]       # 60 features
    ORIGINAL_FEATURES = feat_def["original_features"]           # 46 original
    ENGINEERED_FEATS  = feat_def["engineered_features"]         # 14 engineered
    META_COLS         = feat_def["meta_columns"]
    LABEL_COLS        = feat_def["label_columns"]

    print(f"  Numeric features (S_t dimension): {len(NUMERIC_FEATURES)}")
    print(f"    - Original: {len(ORIGINAL_FEATURES)}")
    print(f"    - Engineered: {len(ENGINEERED_FEATS)}")

    # ── STEP 3: Sort by temporal order ────────────────────────────────────
    print("\n[STEP 3] Sorting dataset by temporal order (day_order, seq_idx)...")
    df = df.sort_values(["day_order", "seq_idx"], ascending=[True, True])
    df = df.reset_index(drop=True)
    print(f"  Sorted. Shape: {df.shape}")

    # Verify day order coverage
    print("\n  Rows per day_order:")
    day_counts = df.groupby("day_order").agg(
        rows=("Label","count"),
        day=("day_name","first"),
        phase=("attack_phase","first"),
        labels=("Label", lambda x: x.value_counts().to_dict())
    )
    for d, row in day_counts.iterrows():
        print(f"    day {d} ({row['day']:<12}) {row['rows']:>8,} rows  |  phase: {row['phase']}")
    report["steps"].append({"step": 3, "action": "sort_temporal"})

    # ── STEP 4: Define S_t — State Vector ────────────────────────────────
    print("\n[STEP 4] Defining State Vector S_t...")

    # S_t is the full 60-dim numeric feature vector + label at time t
    # The label IS part of the state (attack-class trajectory is what we model)
    STATE_FEATURES = NUMERIC_FEATURES  # 60 numeric
    STATE_DIM = len(STATE_FEATURES)
    print(f"  S_t dimension: {STATE_DIM} numeric features")
    print(f"  State features:")
    for i, f in enumerate(STATE_FEATURES):
        tag = " [ENG]" if f in ENGINEERED_FEATS else ""
        print(f"    [{i:3d}] {f}{tag}")

    # ── STEP 5: Build next-state targets S_{t+1} ─────────────────────────
    print("\n[STEP 5] Building next-state targets S_{t+1}...")
    # For each row at position t, the next-state is the row at t+1 (within same day_order)
    # Cross-day boundaries: we mark boundary rows so sequences don't cross days
    
    # Create next-row columns for ALL numeric features + labels
    next_cols = {}
    print("  Creating next_* columns for numeric features and labels...")

    # Shift numeric features by -1 (next row's values)
    for feat in STATE_FEATURES:
        next_cols[f"next_{feat}"] = df[feat].shift(-1)

    # Shift labels by -1 as well (what class does the NEXT flow belong to?)
    next_cols["next_label_binary"]     = df["label_binary"].shift(-1)
    next_cols["next_label_multiclass"] = df["label_multiclass"].shift(-1)
    next_cols["next_Label"]            = df["Label"].shift(-1)

    # Mark boundary rows (last row of each day_order — no valid next state)
    day_last_idx = df.groupby("day_order").tail(1).index
    boundary_mask = df.index.isin(day_last_idx)
    next_cols["is_boundary"] = boundary_mask.astype(int)

    # Create next-state columns at once
    next_df = pd.DataFrame(next_cols, index=df.index)
    df = pd.concat([df, next_df], axis=1)

    boundary_count = int(df["is_boundary"].sum())
    valid_transitions = len(df) - boundary_count
    print(f"  Total rows:          {len(df):,}")
    print(f"  Boundary rows:       {boundary_count}  (last row of each day — no valid S_{{t+1}})")
    print(f"  Valid transitions:   {valid_transitions:,}  (rows with a valid S_{{t+1}})")
    report["steps"].append({
        "step": 5, "action": "build_next_state_targets",
        "total_rows": len(df), "boundary_rows": boundary_count,
        "valid_transitions": valid_transitions
    })

    # ── STEP 6: Chronological train/val/test split ─────────────────────────
    print("\n[STEP 6] Chronological train / val / test split...")
    print(f"  Train: day_order in {TRAIN_DAYS}")
    print(f"  Val:   day_order in {VAL_DAYS}")
    print(f"  Test:  day_order in {TEST_DAYS}")

    df_train = df[df["day_order"].isin(TRAIN_DAYS)].copy()
    df_val   = df[df["day_order"].isin(VAL_DAYS)].copy()
    df_test  = df[df["day_order"].isin(TEST_DAYS)].copy()

    n_total = len(df)
    n_train = len(df_train)
    n_val   = len(df_val)
    n_test  = len(df_test)

    print(f"\n  Split sizes:")
    print(f"    Train: {n_train:>8,}  ({100*n_train/n_total:.1f}%)")
    print(f"    Val:   {n_val:>8,}  ({100*n_val/n_total:.1f}%)")
    print(f"    Test:  {n_test:>8,}  ({100*n_test/n_total:.1f}%)")
    assert n_train + n_val + n_test == n_total, "Split sizes don't add up!"

    print("\n  Train label distribution:")
    tc = df_train["Label"].value_counts()
    for lbl, cnt in tc.items():
        print(f"    {lbl:<35} {cnt:>8,}  ({100*cnt/n_train:.2f}%)")

    print("\n  Val label distribution:")
    vc = df_val["Label"].value_counts()
    for lbl, cnt in vc.items():
        print(f"    {lbl:<35} {cnt:>8,}  ({100*cnt/n_val:.2f}%)")

    print("\n  Test label distribution:")
    tsc = df_test["Label"].value_counts()
    for lbl, cnt in tsc.items():
        print(f"    {lbl:<35} {cnt:>8,}  ({100*cnt/n_test:.2f}%)")

    report["steps"].append({
        "step": 6, "action": "chronological_split",
        "train_days": TRAIN_DAYS, "val_days": VAL_DAYS, "test_days": TEST_DAYS,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "train_label_dist": tc.to_dict(),
        "val_label_dist":   vc.to_dict(),
        "test_label_dist":  tsc.to_dict()
    })

    # ── STEP 7: Verify data integrity ─────────────────────────────────────
    print("\n[STEP 7] Data integrity checks...")
    # No train data should appear in val/test (chronological — guaranteed by day_order)
    train_days_set = set(df_train["day_order"].unique())
    val_days_set   = set(df_val["day_order"].unique())
    test_days_set  = set(df_test["day_order"].unique())
    assert len(train_days_set & val_days_set) == 0,  "Overlap: train/val!"
    assert len(train_days_set & test_days_set) == 0, "Overlap: train/test!"
    assert len(val_days_set   & test_days_set) == 0, "Overlap: val/test!"
    print("  No data leakage between splits. PASSED.")

    # Check temporal ordering within each split
    for name, split_df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        seq = split_df["seq_idx"].values
        # Within each day, seq_idx should be increasing
        for day in split_df["day_order"].unique():
            day_seq = split_df[split_df["day_order"]==day]["seq_idx"].values
            assert (np.diff(day_seq) >= 0).all(), f"{name} day {day}: seq_idx not monotone!"
        print(f"  {name}: temporal ordering verified.")

    # ── STEP 8: Save state datasets ───────────────────────────────────────
    print("\n[STEP 8] Saving state datasets...")

    train_path = PROC_DIR / "state_train.parquet"
    val_path   = PROC_DIR / "state_val.parquet"
    test_path  = PROC_DIR / "state_test.parquet"

    df_train.to_parquet(train_path, index=False, compression="snappy")
    df_val.to_parquet(val_path,     index=False, compression="snappy")
    df_test.to_parquet(test_path,   index=False, compression="snappy")

    for path, label in [(train_path,"Train"),(val_path,"Val"),(test_path,"Test")]:
        size_mb = os.path.getsize(path)/(1024*1024)
        print(f"  {label}: {path.name}  ({size_mb:.1f} MB)")

    report["output_files"] = [str(train_path), str(val_path), str(test_path)]

    # ── STEP 9: Save state definition ─────────────────────────────────────
    print("\n[STEP 9] Saving state definition...")
    state_def = {
        "state_representation": "Flow-level state vector (CIC-IDS2017 MachineLearningCSV)",
        "state_dimension": STATE_DIM,
        "state_features": STATE_FEATURES,
        "original_features": ORIGINAL_FEATURES,
        "engineered_features": ENGINEERED_FEATS,
        "temporal_ordering": {
            "method": "file_row_order_proxy",
            "primary_sort": "day_order (1=Monday, 8=Friday-DDoS)",
            "secondary_sort": "seq_idx (row position within day file)",
            "limitation": "No wall-clock timestamps available. CICFlowMeter strips timestamps. Row order approximates capture order."
        },
        "state_transition": {
            "S_t": "Features of flow at position t",
            "S_t1": "Features of flow at position t+1 (next_* columns)",
            "boundary_handling": "Last row of each day marked is_boundary=1. Sequences must not cross day boundaries.",
            "valid_transitions": valid_transitions
        },
        "split": {
            "strategy": "Chronological (no shuffle) — preserves temporal integrity",
            "train_days": TRAIN_DAYS,
            "val_days": VAL_DAYS,
            "test_days": TEST_DAYS,
            "n_train": n_train, "n_val": n_val, "n_test": n_test,
            "train_pct": round(100*n_train/n_total, 1),
            "val_pct":   round(100*n_val/n_total, 1),
            "test_pct":  round(100*n_test/n_total, 1)
        },
        "label_encoding": label_enc,
        "missing_features_documented": FEATURE_INVENTORY_MISSING,
        "scaling_note": "NO scaling applied in Phase 4. StandardScaler will be FIT ONLY on train split in Phase 6."
    }

    state_def_path = PROC_DIR / "state_definition.json"
    with open(state_def_path, "w") as f:
        json.dump(state_def, f, indent=2)
    print(f"  Saved: {state_def_path}")

    # ── STEP 10: Final summary ────────────────────────────────────────────
    print("\n[STEP 10] Phase 4 Summary...")
    print(f"  State vector S_t : {STATE_DIM}-dimensional")
    print(f"  State features   : {len(ORIGINAL_FEATURES)} original + {len(ENGINEERED_FEATS)} engineered")
    print(f"  Valid transitions: {valid_transitions:,} (rows with S_t → S_{{t+1}})")
    print(f"  Split (chrono):")
    print(f"    Train: {n_train:,} ({100*n_train/n_total:.1f}%)  days {TRAIN_DAYS}")
    print(f"    Val:   {n_val:,} ({100*n_val/n_total:.1f}%)  days {VAL_DAYS}")
    print(f"    Test:  {n_test:,} ({100*n_test/n_total:.1f}%)  days {TEST_DAYS}")

    elapsed = round(time.time()-t0, 2)
    report.update({
        "state_dim": STATE_DIM,
        "state_features": STATE_FEATURES,
        "valid_transitions": valid_transitions,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "elapsed_seconds": elapsed
    })
    rpt_path = METRICS_DIR / "phase4_state_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Report: {rpt_path}")

    print(f"\n{'='*72}")
    print(f"PHASE 4 COMPLETE | Elapsed: {elapsed}s")
    print(f"{'='*72}")
    return df_train, df_val, df_test, state_def

# Missing feature inventory for state definition doc
FEATURE_INVENTORY_MISSING = {
    "Timestamp_wallclock":      "Stripped by CICFlowMeter — row order used as proxy",
    "Source_IP":                "Stripped by CICFlowMeter — host-level states impossible",
    "Destination_IP":           "Stripped by CICFlowMeter",
    "Source_Port":              "Stripped by CICFlowMeter",
    "IP_TTL":                   "PCAP-level only",
    "TCP_Retransmission_count": "Not in CICFlowMeter output",
    "Payload_content":          "Raw PCAP required",
    "Per_packet_IAT":           "Only flow-aggregate IAT available"
}

if __name__ == "__main__":
    df_train, df_val, df_test, state_def = main()
    print(f"\nState definition saved.")
    print(f"Train columns ({len(df_train.columns)}): first 10 = {list(df_train.columns[:10])}")
