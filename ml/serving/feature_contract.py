import json
from pathlib import Path

# Load from the true source of truth
PROC_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
try:
    with open(PROC_DIR / "feature_list.json") as f:
        _feat_def = json.load(f)
except FileNotFoundError:
    _feat_def = {"all_numeric_features": [], "original_features": []}

REQUIRED_MODEL_FEATURES = _feat_def.get("all_numeric_features", [])
ORIGINAL_FEATURES       = _feat_def.get("original_features", [])

# A safe subset of 20 core flow characteristics that are almost always present in any raw network CSV/Parquet.
PARTIAL_MODEL_FEATURES = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Fwd IAT Total",
    "Bwd IAT Total",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
]

# We must ensure that PARTIAL_MODEL_FEATURES are actually in ORIGINAL_FEATURES
# so we don't accidentally invent a feature that the training dataset never had.
PARTIAL_MODEL_FEATURES = [f for f in PARTIAL_MODEL_FEATURES if f in ORIGINAL_FEATURES]
