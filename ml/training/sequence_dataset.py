"""
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
