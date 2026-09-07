import pandas as pd
import numpy as np
from ml.serving.feature_contract import REQUIRED_MODEL_FEATURES

def safe_col(df: pd.DataFrame, col: str) -> pd.Series:
    """Returns the column if it exists, otherwise returns a Series of zeros."""
    if col in df.columns:
        return df[col]
    return pd.Series(0.0, index=df.index, dtype=np.float64)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministically reconstructs the 14 engineered features from the original raw features.
    This exactly mirrors the logic in ml/preprocessing/phase3_features.py.
    """
    df_out = df.copy()

    # Ratio features
    df_out["pkt_fwd_bwd_ratio"]    = safe_col(df_out, "Total Fwd Packets") / (safe_col(df_out, "Total Backward Packets") + 1.0)
    df_out["byte_fwd_bwd_ratio"]   = safe_col(df_out, "Total Length of Fwd Packets") / (safe_col(df_out, "Total Length of Bwd Packets") + 1.0)
    df_out["avg_pkt_size_ratio"]   = safe_col(df_out, "Fwd Packet Length Mean") / (safe_col(df_out, "Bwd Packet Length Mean") + 1.0)
    df_out["fwd_bwd_iat_ratio"]    = safe_col(df_out, "Fwd IAT Mean") / (safe_col(df_out, "Bwd IAT Mean") + 1.0)
    df_out["active_idle_ratio"]    = safe_col(df_out, "Active Mean") / (safe_col(df_out, "Idle Mean") + 1.0)
    df_out["syn_fin_ratio"]        = safe_col(df_out, "SYN Flag Count") / (safe_col(df_out, "FIN Flag Count") + 1.0)
    df_out["window_size_ratio"]    = safe_col(df_out, "Init_Win_bytes_forward") / (safe_col(df_out, "Init_Win_bytes_backward") + 1.0)
    df_out["total_packets"]        = safe_col(df_out, "Total Fwd Packets") + safe_col(df_out, "Total Backward Packets")
    df_out["total_bytes"]          = safe_col(df_out, "Total Length of Fwd Packets") + safe_col(df_out, "Total Length of Bwd Packets")

    # Flag density
    flag_cols = [c for c in ["FIN Flag Count","SYN Flag Count","RST Flag Count",
                               "PSH Flag Count","ACK Flag Count","URG Flag Count"] if c in df_out.columns]
    if flag_cols:
        df_out["flag_density_per_pkt"] = df_out[flag_cols].sum(axis=1) / (df_out["total_packets"] + 1.0)
    else:
        df_out["flag_density_per_pkt"] = 0.0

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
        if src_col in df_out.columns:
            # We clip at 0 to avoid log(-negative) warning
            df_out[new_col] = np.log1p(df_out[src_col].clip(lower=0))
            applied_logs.append(new_col)
        else:
            df_out[new_col] = 0.0
            applied_logs.append(new_col)

    NEW_FEATURES = ["pkt_fwd_bwd_ratio","byte_fwd_bwd_ratio","avg_pkt_size_ratio",
                    "fwd_bwd_iat_ratio","active_idle_ratio","syn_fin_ratio",
                    "window_size_ratio","total_packets","total_bytes",
                    "flag_density_per_pkt"] + applied_logs

    # Handle NaN/Inf by filling with 0.0
    for col in NEW_FEATURES:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace([np.inf, -np.inf], np.nan)
            df_out[col] = df_out[col].fillna(0.0)

    return df_out
