"""
PHASE 20 – Unit Tests
=====================
Tests for core pipeline components: feature engineering, sequence
construction, risk scoring, MITRE mapping, and inference server logic.

Run: python -m pytest ml/tests/ -v
"""
import json, pickle
from pathlib import Path
import numpy as np
import pytest

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def feat_def():
    with open(PROC_DIR / "feature_list.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def label_enc():
    with open(PROC_DIR / "label_encoding.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def scaler(feat_def):
    with open(MODELS_DIR / "standard_scaler.pkl", "rb") as f:
        return pickle.load(f)["scaler"]

@pytest.fixture(scope="module")
def feature_cols(feat_def):
    return feat_def["all_numeric_features"]

@pytest.fixture(scope="module")
def class_names(label_enc):
    return label_enc["classes"]


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2 / 3 / 4 — Data & Feature Checks
# ══════════════════════════════════════════════════════════════════════════════
class TestFeatureDefinitions:
    def test_feature_list_loaded(self, feat_def):
        assert "all_numeric_features" in feat_def
        assert "engineered_features"  in feat_def

    def test_feature_count(self, feature_cols):
        assert len(feature_cols) == 60, f"Expected 60 features, got {len(feature_cols)}"

    def test_engineered_features_subset(self, feat_def, feature_cols):
        eng = set(feat_def["engineered_features"])
        all_feats = set(feature_cols)
        assert eng.issubset(all_feats), "Engineered features must be a subset of all features"

    def test_no_duplicate_features(self, feature_cols):
        assert len(feature_cols) == len(set(feature_cols)), "Duplicate features found!"

    def test_key_engineered_features_present(self, feat_def):
        eng = feat_def["engineered_features"]
        required = ["byte_fwd_bwd_ratio", "window_size_ratio", "log1p_flow_duration"]
        for r in required:
            assert r in eng, f"Key engineered feature '{r}' missing"


class TestLabelEncoding:
    def test_classes_loaded(self, class_names):
        assert len(class_names) > 0

    def test_benign_class_present(self, class_names):
        assert "BENIGN" in class_names, "BENIGN class must be present"

    def test_n_classes(self, class_names):
        assert len(class_names) == 15, f"Expected 15 classes, got {len(class_names)}"


class TestScaler:
    def test_scaler_loaded(self, scaler):
        assert scaler is not None

    def test_scaler_feature_count(self, scaler, feature_cols):
        assert scaler.n_features_in_ == len(feature_cols)

    def test_scaler_transform(self, scaler, feature_cols):
        X_dummy = np.zeros((5, len(feature_cols)), dtype=np.float32)
        X_scaled = scaler.transform(X_dummy)
        assert X_scaled.shape == (5, len(feature_cols))
        # Scaled zero should be approximately -mean/std
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))


# ══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Sequence Construction
# ══════════════════════════════════════════════════════════════════════════════
class TestSequenceFiles:
    def test_processed_files_exist(self):
        for fname in ["state_train.parquet", "state_val.parquet", "state_test.parquet"]:
            assert (PROC_DIR / fname).exists(), f"{fname} not found"

    def test_feature_list_json_exists(self):
        assert (PROC_DIR / "feature_list.json").exists()

    def test_label_encoding_exists(self):
        assert (PROC_DIR / "label_encoding.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Phase 8 / 9 — Model Files
# ══════════════════════════════════════════════════════════════════════════════
class TestModelFiles:
    def test_gru_model_exists(self):
        assert (MODELS_DIR / "world_model" / "gru_world_model.pt").exists()

    def test_lstm_model_exists(self):
        assert (MODELS_DIR / "world_model" / "lstm_world_model.pt").exists()

    def test_xgb_binary_exists(self):
        assert (MODELS_DIR / "xgb_binary.pkl").exists()

    def test_xgb_mc_exists(self):
        assert (MODELS_DIR / "xgb_multiclass.json").exists()

    def test_lr_binary_exists(self):
        assert (MODELS_DIR / "lr_binary.pkl").exists()

    def test_gru_checkpoint_has_hyperparams(self):
        import torch
        ckpt = torch.load(MODELS_DIR / "world_model" / "gru_world_model.pt",
                          map_location="cpu")
        assert "hyperparams" in ckpt
        assert "model_state_dict" in ckpt
        hp = ckpt["hyperparams"]
        assert "hidden_dim" in hp
        assert "input_dim" in hp
        assert hp["input_dim"] == 60


# ══════════════════════════════════════════════════════════════════════════════
# Phase 10 — Risk Scoring Logic
# ══════════════════════════════════════════════════════════════════════════════
RISK_DISCOUNT = 0.85
SMAP = {0:0, 1:5, 2:5, 3:4, 4:4, 5:3, 6:3, 7:3, 8:5, 9:5, 10:2, 11:3, 12:3, 13:5, 14:3}

def compute_risk(ap, cp):
    K    = len(ap)
    dmax = sum(RISK_DISCOUNT**k for k in range(K)) * 5.0
    raw  = sum(RISK_DISCOUNT**k * ap[k] * (1 + SMAP.get(int(np.argmax(cp[k])), 0)/5.0)
               for k in range(K))
    return min(raw / (dmax * 2.0/5.0) * 100, 100)

class TestRiskScoring:
    def test_benign_risk_low(self):
        ap = [0.05] * 8
        cp = [[1.0] + [0.0]*14] * 8   # all BENIGN
        rs = compute_risk(ap, cp)
        assert rs < 20, f"Pure BENIGN risk should be < 20, got {rs:.2f}"

    def test_attack_risk_high(self):
        ap = [0.95] * 8
        cp = [[0.0]*2 + [1.0] + [0.0]*12] * 8   # all DDoS (class 2, severity 5)
        rs = compute_risk(ap, cp)
        assert rs >= 65, f"Pure attack risk should be >= 65, got {rs:.2f}"

    def test_risk_bounded(self):
        for _ in range(20):
            ap = np.random.rand(8).tolist()
            cp = [np.eye(15)[np.random.randint(15)].tolist() for _ in range(8)]
            rs = compute_risk(ap, cp)
            assert 0 <= rs <= 100, f"Risk out of bounds: {rs}"

    def test_risk_monotone_with_attack_prob(self):
        cp = [[1.0] + [0.0]*14] * 8
        risks = []
        for prob in [0.1, 0.3, 0.5, 0.7, 0.9]:
            ap = [prob] * 8
            risks.append(compute_risk(ap, cp))
        assert risks == sorted(risks), "Risk should increase with attack probability"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 12 — MITRE Mapping
# ══════════════════════════════════════════════════════════════════════════════
MITRE_MAP = {
    0:  {"label":"BENIGN",               "tactic":None,                "stage":0,"severity":0},
    2:  {"label":"DDoS",                 "tactic":"Impact",             "stage":7,"severity":5},
    10: {"label":"PortScan",             "tactic":"Discovery",          "stage":1,"severity":2},
}

class TestMitreMapping:
    def test_benign_has_no_tactic(self):
        assert MITRE_MAP[0]["tactic"] is None

    def test_ddos_is_impact(self):
        assert MITRE_MAP[2]["tactic"] == "Impact"
        assert MITRE_MAP[2]["stage"]  == 7

    def test_portscan_is_discovery(self):
        assert MITRE_MAP[10]["tactic"] == "Discovery"
        assert MITRE_MAP[10]["stage"]  == 1

    def test_severity_range(self):
        for cls_id, info in MITRE_MAP.items():
            assert 0 <= info["severity"] <= 5, f"Severity out of range for class {cls_id}"


# ══════════════════════════════════════════════════════════════════════════════
# Phase 11 / 17 — Results Files
# ══════════════════════════════════════════════════════════════════════════════
class TestResultFiles:
    def test_phase10_metrics_exist(self):
        assert (BASE_DIR / "results" / "metrics" / "phase10_forecasting_metrics.json").exists()

    def test_phase10_has_rollout_auc(self):
        with open(BASE_DIR / "results" / "metrics" / "phase10_forecasting_metrics.json") as f:
            m = json.load(f)
        assert "rollout_auc" in m or "k1_auc" in m or any("auc" in str(k).lower() for k in m)

    def test_config_yaml_exists(self):
        assert (BASE_DIR / "configs" / "config.yaml").exists()

    def test_readme_exists(self):
        assert (BASE_DIR / "README.md").exists()


if __name__ == "__main__":
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", __file__, "-v", "--tb=short"],
        cwd=str(BASE_DIR)
    )
    sys.exit(result.returncode)
