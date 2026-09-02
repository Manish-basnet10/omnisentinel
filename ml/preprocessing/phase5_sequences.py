"""
PHASE 5 – Temporal Sequence Construction
==========================================
SIH-2026 | CIC-IDS2017 | World Model Pipeline

DESIGN
------
Sliding window over the chronologically sorted state vectors:
  Input  : [S_{t-L+1}, S_{t-L+2}, ..., S_t]   shape (L, D)
  Target : S_{t+1}                              shape (D,)   ← next-state regression
           label_binary_{t+1}                  scalar (0/1) ← attack binary classification
           label_multiclass_{t+1}              scalar (int) ← 15-class classification

Parameters:
  SEQ_LEN (L)  = 10   (10 consecutive flows as temporal context window)
  STRIDE        = 5    (step between windows in train)  
  STRIDE_EVAL   = 1    (step for val/test)

Boundary Rule:
  A sequence is VALID only if all L+1 rows (L input + 1 target) belong to
  the SAME day_order. This prevents learning spurious transitions across days.

Outputs (production-grade — index-based for train, numpy for val/test):
  data/processed/sequences/
    seq_indices_train.npy   — start row indices for training sequences
    seq_indices_val.npy     — start row indices for validation sequences
    seq_indices_test.npy    — start row indices for test sequences
    X_val.npy               — (N_val, L, D)  float32
    y_val_state.npy         — (N_val, D)     float32  next-state
    y_val_binary.npy        — (N_val,)       int8     next label binary
    y_val_multiclass.npy    — (N_val,)       int8     next label multiclass
    X_test.npy              — (N_test, L, D) float32
    y_test_state.npy        — (N_test, D)    float32
    y_test_binary.npy       — (N_test,)      int8
    y_test_multiclass.npy   — (N_test,)      int8
    sequence_config.json    — all parameters + feature list
    sequence_stats.json     — counts, memory estimates, boundary stats

ml/training/
    sequence_dataset.py     — PyTorch Dataset (index-based, memory-efficient for train)

STRICT RULES:
  - Sequences NEVER cross day boundaries
  - StandardScaler NOT applied here (done in Phase 6 on train only)
  - Fixed seed = 42
  - All counts are real, no fabrication
"""

import os, json, warnings, gc, time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED      = 42
SEQ_LEN   = 10          # L: temporal context window
STRIDE    = 5           # step between windows (train)
STRIDE_EVAL = 1         # step for val/test

BASE_DIR  = Path("/Users/manishbasnet/Desktop/omnisentinel/ai-network-attack-forecasting")
PROC_DIR  = BASE_DIR / "data" / "processed"
SEQ_DIR   = PROC_DIR / "sequences"
METRICS_DIR = BASE_DIR / "results" / "metrics"
SEQ_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
def build_sequence_indices(df_sorted, seq_len, stride):
    """
    Compute valid sequence START indices within df_sorted.
    A sequence starting at index i uses rows i..i+seq_len (inclusive).
    Valid = all rows in same day_order AND none is a boundary row.

    Returns: np.array of valid start indices (into df_sorted's integer index)
    """
    day_orders   = df_sorted["day_order"].values
    is_boundary  = df_sorted["is_boundary"].values
    n = len(df_sorted)

    valid_starts = []
    i = 0
    while i + seq_len < n:          # need seq_len input + 1 target row
        end = i + seq_len           # target row index (inclusive)
        # Check: all seq_len+1 rows must have same day_order
        if day_orders[i] != day_orders[end]:
            # Skip to next day start
            i += 1
            continue
        # Check: none of the input rows (i..end-1) is a boundary
        if np.any(is_boundary[i:end]):
            i += 1
            continue
        # Check: the target row (end) is not a boundary (it has a valid next row)
        # Actually we just need the target row's features — boundary of target is OK
        # as long as the sequence itself doesn't span a boundary
        valid_starts.append(i)
        i += stride

    return np.array(valid_starts, dtype=np.int64)


def build_numpy_sequences(df_sorted, indices, feature_cols, seq_len):
    """
    Build X (input sequences) and y_* (targets) as numpy arrays.
    X shape: (N, seq_len, n_features)
    """
    n_seq = len(indices)
    n_feat = len(feature_cols)
    feat_values = df_sorted[feature_cols].values.astype(np.float32)
    y_bin_vals  = df_sorted["next_label_binary"].values
    y_mc_vals   = df_sorted["next_label_multiclass"].values
    # Build next-state target from next_* columns
    next_feat_cols = [f"next_{c}" for c in feature_cols]
    y_state_vals = df_sorted[next_feat_cols].values.astype(np.float32)

    X       = np.empty((n_seq, seq_len, n_feat), dtype=np.float32)
    y_state = np.empty((n_seq, n_feat),           dtype=np.float32)
    y_bin   = np.empty((n_seq,),                  dtype=np.int8)
    y_mc    = np.empty((n_seq,),                  dtype=np.int8)

    for j, start in enumerate(indices):
        end = start + seq_len          # target row
        X[j]       = feat_values[start:end]       # rows start..end-1
        y_state[j] = y_state_vals[end - 1]        # S_{t+1} = next_* of last input row
        y_bin[j]   = y_bin_vals[end - 1]
        y_mc[j]    = y_mc_vals[end - 1]
        if j % 50000 == 0 and j > 0:
            print(f"    [{j:>7,}/{n_seq:,}] built...", flush=True)

    return X, y_state, y_bin, y_mc


def main():
    t0 = time.time()
    print("="*72)
    print("PHASE 5 – TEMPORAL SEQUENCE CONSTRUCTION")
    print("="*72)
    print(f"  SEQ_LEN={SEQ_LEN}  STRIDE_TRAIN={STRIDE}  STRIDE_EVAL={STRIDE_EVAL}")

    report = {"phase":5, "seq_len":SEQ_LEN, "stride_train":STRIDE,
              "stride_eval":STRIDE_EVAL, "steps":[]}

    # ── STEP 1: Load feature list & splits ────────────────────────────────
    print("\n[STEP 1] Loading state splits and feature list...")
    with open(PROC_DIR/"feature_list.json") as f:
        feat_def = json.load(f)
    with open(PROC_DIR/"label_encoding.json") as f:
        label_enc = json.load(f)

    FEATURE_COLS = feat_def["all_numeric_features"]   # 60 features = S_t
    D = len(FEATURE_COLS)
    print(f"  Feature dimension D = {D}")

    df_train = pd.read_parquet(PROC_DIR/"state_train.parquet")
    df_val   = pd.read_parquet(PROC_DIR/"state_val.parquet")
    df_test  = pd.read_parquet(PROC_DIR/"state_test.parquet")
    print(f"  Train: {len(df_train):,}  Val: {len(df_val):,}  Test: {len(df_test):,}")

    # Verify next_* columns exist (built in Phase 4)
    next_cols = [f"next_{c}" for c in FEATURE_COLS]
    for c in next_cols[:3]:
        assert c in df_train.columns, f"Missing {c}"
    assert "next_label_binary"     in df_train.columns
    assert "next_label_multiclass" in df_train.columns
    print("  next_* columns verified.")
    report["steps"].append({"step":1,"D":D,"n_train":len(df_train),
                             "n_val":len(df_val),"n_test":len(df_test)})

    # ── STEP 2: Verify boundary markers ───────────────────────────────────
    print("\n[STEP 2] Boundary markers check...")
    for name, df in [("Train",df_train),("Val",df_val),("Test",df_test)]:
        b = int(df["is_boundary"].sum())
        print(f"  {name}: {b} boundary rows (last row of each day)")
    report["steps"].append({"step":2,"action":"boundary_check"})

    # ── STEP 3: Build TRAIN sequence indices (index-based, memory-efficient)
    print(f"\n[STEP 3] Building TRAIN sequence indices (L={SEQ_LEN}, stride={STRIDE})...")
    # Sort by day_order, seq_idx
    df_train = df_train.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    idx_train = build_sequence_indices(df_train, SEQ_LEN, STRIDE)
    n_train_seq = len(idx_train)
    print(f"  Valid train sequences: {n_train_seq:,}")

    train_idx_path = SEQ_DIR / "seq_indices_train.npy"
    np.save(train_idx_path, idx_train)
    print(f"  Saved: {train_idx_path}  ({os.path.getsize(train_idx_path)/1024:.1f} KB)")

    # Memory estimate for full numpy build (if done)
    mem_est_gb = n_train_seq * SEQ_LEN * D * 4 / (1024**3)
    print(f"  (Full X_train numpy would be ≈ {mem_est_gb:.2f} GB — using index-based Dataset instead)")
    report["steps"].append({"step":3,"n_train_seq":int(n_train_seq),
                             "mem_est_X_train_gb":round(mem_est_gb,2)})

    # ── STEP 4: Build VAL numpy arrays ────────────────────────────────────
    print(f"\n[STEP 4] Building VAL numpy arrays (L={SEQ_LEN}, stride={STRIDE_EVAL})...")
    df_val = df_val.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    idx_val = build_sequence_indices(df_val, SEQ_LEN, STRIDE_EVAL)
    n_val_seq = len(idx_val)
    print(f"  Valid val sequences: {n_val_seq:,}")

    print("  Building X_val, y_val arrays...")
    X_val, y_val_state, y_val_bin, y_val_mc = build_numpy_sequences(
        df_val, idx_val, FEATURE_COLS, SEQ_LEN)

    val_idx_path   = SEQ_DIR / "seq_indices_val.npy"
    X_val_path     = SEQ_DIR / "X_val.npy"
    y_val_s_path   = SEQ_DIR / "y_val_state.npy"
    y_val_b_path   = SEQ_DIR / "y_val_binary.npy"
    y_val_mc_path  = SEQ_DIR / "y_val_multiclass.npy"

    np.save(val_idx_path,  idx_val)
    np.save(X_val_path,    X_val)
    np.save(y_val_s_path,  y_val_state)
    np.save(y_val_b_path,  y_val_bin)
    np.save(y_val_mc_path, y_val_mc)

    print(f"  X_val shape: {X_val.shape}  dtype={X_val.dtype}")
    for p in [X_val_path, y_val_s_path, y_val_b_path, y_val_mc_path]:
        print(f"    Saved: {p.name}  ({os.path.getsize(p)/1024:.0f} KB)")

    print(f"  Val binary distribution:  0={int((y_val_bin==0).sum()):,}  1={int((y_val_bin==1).sum()):,}")
    val_mc_unique, val_mc_counts = np.unique(y_val_mc, return_counts=True)
    print(f"  Val multiclass: classes={val_mc_unique.tolist()}")

    report["steps"].append({"step":4,"n_val_seq":int(n_val_seq),
                             "X_val_shape":list(X_val.shape)})
    del X_val, y_val_state; gc.collect()

    # ── STEP 5: Build TEST numpy arrays ───────────────────────────────────
    print(f"\n[STEP 5] Building TEST numpy arrays (L={SEQ_LEN}, stride={STRIDE_EVAL})...")
    df_test = df_test.sort_values(["day_order","seq_idx"]).reset_index(drop=True)
    idx_test = build_sequence_indices(df_test, SEQ_LEN, STRIDE_EVAL)
    n_test_seq = len(idx_test)
    print(f"  Valid test sequences: {n_test_seq:,}")

    print("  Building X_test, y_test arrays...")
    X_test, y_test_state, y_test_bin, y_test_mc = build_numpy_sequences(
        df_test, idx_test, FEATURE_COLS, SEQ_LEN)

    test_idx_path   = SEQ_DIR / "seq_indices_test.npy"
    X_test_path     = SEQ_DIR / "X_test.npy"
    y_test_s_path   = SEQ_DIR / "y_test_state.npy"
    y_test_b_path   = SEQ_DIR / "y_test_binary.npy"
    y_test_mc_path  = SEQ_DIR / "y_test_multiclass.npy"

    np.save(test_idx_path,  idx_test)
    np.save(X_test_path,    X_test)
    np.save(y_test_s_path,  y_test_state)
    np.save(y_test_b_path,  y_test_bin)
    np.save(y_test_mc_path, y_test_mc)

    print(f"  X_test shape: {X_test.shape}  dtype={X_test.dtype}")
    for p in [X_test_path, y_test_s_path, y_test_b_path, y_test_mc_path]:
        print(f"    Saved: {p.name}  ({os.path.getsize(p)/(1024*1024):.1f} MB)")

    print(f"  Test binary distribution: 0={int((y_test_bin==0).sum()):,}  1={int((y_test_bin==1).sum()):,}")
    test_mc_unique, test_mc_counts = np.unique(y_test_mc, return_counts=True)
    print(f"  Test multiclass: classes={test_mc_unique.tolist()}  counts={test_mc_counts.tolist()}")

    report["steps"].append({"step":5,"n_test_seq":int(n_test_seq),
                             "X_test_shape":list(X_test.shape),
                             "test_binary_dist":{"0":int((y_test_bin==0).sum()),
                                                 "1":int((y_test_bin==1).sum())}})
    del X_test, y_test_state; gc.collect()

    # ── STEP 6: Save sequence config ─────────────────────────────────────
    print("\n[STEP 6] Saving sequence config...")
    seq_config = {
        "seq_len": SEQ_LEN,
        "stride_train": STRIDE,
        "stride_eval": STRIDE_EVAL,
        "feature_dim": D,
        "feature_cols": FEATURE_COLS,
        "n_classes_binary": 2,
        "n_classes_multiclass": len(label_enc["classes"]),
        "class_names": label_enc["classes"],
        "class2idx": label_enc["class2idx"],
        "n_seq_train": int(n_train_seq),
        "n_seq_val": int(n_val_seq),
        "n_seq_test": int(n_test_seq),
        "seq_dir": str(SEQ_DIR),
        "boundary_rule": "Sequences never cross day_order boundaries",
        "temporal_ordering": "file_row_order_proxy (no wall-clock timestamps)",
        "scaling_note": "NO scaling applied. StandardScaler fit on train in Phase 6.",
        "train_indices_file": str(train_idx_path),
        "val_X_file": str(X_val_path),
        "test_X_file": str(X_test_path),
        "target_tasks": {
            "world_model_regression": "Predict next state S_{t+1} (D-dim regression)",
            "binary_classification": "Predict next flow is attack (0/1)",
            "multiclass_classification": "Predict next flow attack class (15 classes)"
        }
    }
    cfg_path = SEQ_DIR / "sequence_config.json"
    with open(cfg_path, "w") as f:
        json.dump(seq_config, f, indent=2)
    print(f"  Saved: {cfg_path}")

    # ── STEP 7: Sequence statistics ───────────────────────────────────────
    print("\n[STEP 7] Sequence statistics summary:")
    print(f"  {'Split':<10} {'Sequences':>12}  {'Input shape':>15}  {'Attack seqs':>12}")
    print(f"  {'-'*55}")
    attack_train_est = "N/A (index-based)"
    print(f"  {'Train':<10} {n_train_seq:>12,}  {f'({SEQ_LEN},{D})':>15}  {attack_train_est:>12}")
    n_attack_val  = int((y_val_bin==1).sum())
    n_attack_test = int((y_test_bin==1).sum())
    print(f"  {'Val':<10} {n_val_seq:>12,}  {f'({SEQ_LEN},{D})':>15}  {n_attack_val:>12,}")
    print(f"  {'Test':<10} {n_test_seq:>12,}  {f'({SEQ_LEN},{D})':>15}  {n_attack_test:>12,}")

    seq_stats = {
        "n_seq_train": int(n_train_seq),
        "n_seq_val": int(n_val_seq),
        "n_seq_test": int(n_test_seq),
        "total_sequences": int(n_train_seq + n_val_seq + n_test_seq),
        "attack_seqs_val":  int(n_attack_val),
        "attack_seqs_test": int(n_attack_test),
        "attack_pct_val":  round(100*n_attack_val/n_val_seq, 2),
        "attack_pct_test": round(100*n_attack_test/n_test_seq, 2),
        "mem_est_X_train_gb": round(mem_est_gb, 2),
        "mem_X_val_mb":  round(os.path.getsize(X_val_path)/(1024**2), 1),
        "mem_X_test_mb": round(os.path.getsize(X_test_path)/(1024**2), 1),
    }
    stats_path = SEQ_DIR / "sequence_stats.json"
    with open(stats_path, "w") as f:
        json.dump(seq_stats, f, indent=2)
    print(f"\n  Stats saved: {stats_path}")

    # ── STEP 8: Write PyTorch Dataset class ──────────────────────────────
    print("\n[STEP 8] Writing PyTorch Dataset class...")
    dataset_code = '''"""
NetworkSequenceDataset — Memory-efficient PyTorch Dataset for World Model training.
Phase 5 output. Used in Phase 8 (GRU) and Phase 9 (LSTM).

Usage:
    from ml.training.sequence_dataset import NetworkSequenceDataset
    dataset = NetworkSequenceDataset(df_train, seq_len=10, feature_cols=FEATURE_COLS, scaler=scaler)
    loader  = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class NetworkSequenceDataset(Dataset):
    """
    Index-based temporal sequence dataset.

    For each valid sequence start index i:
      X[i]  = scaled feature matrix  rows[i : i+seq_len]        shape (seq_len, D)
      y_state[i] = scaled next-state  next_* values of row[i+seq_len-1]  shape (D,)
      y_bin[i]   = next_label_binary  of row[i+seq_len-1]       scalar int
      y_mc[i]    = next_label_multiclass of row[i+seq_len-1]    scalar int

    Boundary rule: Sequences never cross day_order boundaries.
    Scaler: StandardScaler fitted ONLY on training data (Phase 6).
    """

    def __init__(self, df: pd.DataFrame, seq_len: int, feature_cols: list,
                 scaler=None, stride: int = 1):
        """
        Args:
            df          : Sorted DataFrame (by day_order, seq_idx) with
                          feature_cols, next_* cols, label_binary, label_multiclass
            seq_len     : Number of time steps L in each sequence
            feature_cols: List of feature column names (D features)
            scaler      : Fitted sklearn StandardScaler (None = no scaling)
            stride      : Step between sequence windows
        """
        super().__init__()
        self.seq_len      = seq_len
        self.feature_cols = feature_cols
        self.D            = len(feature_cols)
        self.next_cols    = [f"next_{c}" for c in feature_cols]

        # Extract numpy arrays (raw values — scaler applied below)
        feat_arr       = df[feature_cols].values.astype(np.float32)
        next_feat_arr  = df[self.next_cols].values.astype(np.float32)
        self.y_bin_arr = df["next_label_binary"].values.astype(np.int64)
        self.y_mc_arr  = df["next_label_multiclass"].values.astype(np.int64)

        # Apply scaler if provided
        if scaler is not None:
            feat_arr      = scaler.transform(feat_arr).astype(np.float32)
            next_feat_arr = scaler.transform(next_feat_arr).astype(np.float32)

        self.feat_arr      = feat_arr
        self.next_feat_arr = next_feat_arr

        # Build valid start indices
        day_orders  = df["day_order"].values
        is_boundary = df["is_boundary"].values
        n = len(df)

        valid_starts = []
        i = 0
        while i + seq_len < n:
            end = i + seq_len
            if day_orders[i] != day_orders[end]:
                i += 1; continue
            if np.any(is_boundary[i:end]):
                i += 1; continue
            valid_starts.append(i)
            i += stride

        self.indices = np.array(valid_starts, dtype=np.int64)
        print(f"  NetworkSequenceDataset: {len(self.indices):,} sequences "
              f"(L={seq_len}, stride={stride}, D={self.D})")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start  = self.indices[idx]
        end    = start + self.seq_len          # target row index
        X      = torch.from_numpy(self.feat_arr[start:end])            # (L, D)
        y_state= torch.from_numpy(self.next_feat_arr[end - 1])         # (D,)
        y_bin  = torch.tensor(self.y_bin_arr[end - 1], dtype=torch.long)
        y_mc   = torch.tensor(self.y_mc_arr[end - 1],  dtype=torch.long)
        return X, y_state, y_bin, y_mc


class SequenceIndexDataset(Dataset):
    """
    Lightweight variant: pre-loaded index array + pre-scaled numpy arrays.
    Faster for very large training sets — avoids rebuilding indices.
    """
    def __init__(self, feat_arr, next_feat_arr, y_bin_arr, y_mc_arr,
                 indices, seq_len):
        self.feat_arr      = feat_arr
        self.next_feat_arr = next_feat_arr
        self.y_bin_arr     = y_bin_arr
        self.y_mc_arr      = y_mc_arr
        self.indices       = indices
        self.seq_len       = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.seq_len
        X       = torch.from_numpy(self.feat_arr[start:end].copy())
        y_state = torch.from_numpy(self.next_feat_arr[end - 1].copy())
        y_bin   = torch.tensor(int(self.y_bin_arr[end - 1]), dtype=torch.long)
        y_mc    = torch.tensor(int(self.y_mc_arr[end - 1]),  dtype=torch.long)
        return X, y_state, y_bin, y_mc
'''
    ds_path = BASE_DIR / "ml" / "training" / "sequence_dataset.py"
    ds_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ds_path, "w") as f:
        f.write(dataset_code)
    print(f"  Saved: {ds_path}")

    # ── Finalize report ───────────────────────────────────────────────────
    elapsed = round(time.time()-t0, 2)
    report.update({
        "n_seq_train": int(n_train_seq),
        "n_seq_val":   int(n_val_seq),
        "n_seq_test":  int(n_test_seq),
        "total_sequences": int(n_train_seq + n_val_seq + n_test_seq),
        "elapsed_seconds": elapsed,
        "sequence_stats": seq_stats
    })
    rpt_path = METRICS_DIR / "phase5_sequence_report.json"
    with open(rpt_path,"w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n{'='*72}")
    print(f"PHASE 5 COMPLETE | Elapsed: {elapsed}s")
    print(f"  SEQ_LEN={SEQ_LEN}  D={D}")
    print(f"  Train sequences : {n_train_seq:,}  (index-based — memory-efficient)")
    print(f"  Val sequences   : {n_val_seq:,}    (numpy arrays saved)")
    print(f"  Test sequences  : {n_test_seq:,}   (numpy arrays saved)")
    print(f"  Total sequences : {n_train_seq+n_val_seq+n_test_seq:,}")
    print(f"  Output dir: {SEQ_DIR}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
