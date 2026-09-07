import re
from pathlib import Path

file_path = Path("ml/serving/inference_server.py")
content = file_path.read_text()

imports = """
from ml.models.partial_model import PartialFlowMLP
from ml.serving.feature_contract import PARTIAL_MODEL_FEATURES, ORIGINAL_FEATURES
from ml.preprocessing.inference_engineer import engineer_features
import joblib
"""
content = content.replace("import torch.nn.functional as F", "import torch.nn.functional as F\n" + imports)

# Global variables
globals_block = """_xgb_binary = None   # XGBoost binary classifier (for PCAP per-flow analysis)
_feat_extractor = None  # PCAPFeatureExtractor (lazy-init)
_partial_model = None
_partial_scaler = None
"""
content = content.replace("_xgb_binary = None   # XGBoost binary classifier (for PCAP per-flow analysis)\n_feat_extractor = None  # PCAPFeatureExtractor (lazy-init)", globals_block)

load_block = """def load_model():
    global _model, _scaler, _feat_cols, _class_names, _eng_feats, _device, _demo_preds
    global _xgb_binary, _feat_extractor, _partial_model, _partial_scaler"""
content = content.replace("def load_model():\n    global _model, _scaler, _feat_cols, _class_names, _eng_feats, _device, _demo_preds\n    global _xgb_binary, _feat_extractor", load_block)

load_partial = """
    # Load Partial PyTorch model if exists
    partial_path = MODELS_DIR / "partial_model.pth"
    scaler_path = MODELS_DIR / "partial_scaler.pkl"
    if partial_path.exists() and scaler_path.exists():
        _partial_scaler = joblib.load(scaler_path)
        _partial_model = PartialFlowMLP(input_size=len(PARTIAL_MODEL_FEATURES), num_classes=15).to(device)
        _partial_model.load_state_dict(torch.load(partial_path, map_location=device))
        _partial_model.eval()
        logger.info("[SERVER] Partial PyTorch Model loaded successfully.")
    else:
        logger.warning("[SERVER] Partial model or scaler not found. Fallback mode will be unavailable.")

    # Load XGBoost binary model (for per-flow PCAP classification)"""
content = content.replace("    # Load XGBoost binary model (for per-flow PCAP classification)", load_partial)

# Replace _analyze_file_sync
start_idx = content.find("def _analyze_file_sync(")
end_idx = content.find("\n@app.post(\"/api/analyze-pcap\")")

new_func = """def _analyze_file_sync(file_path: Path, original_filename: str) -> dict:
    t_start = time.perf_counter()
    work_dir = None
    suffix = file_path.suffix.lower()

    feature_coverage = {
        "required": 60,
        "available": 0,
        "engineered": 0,
        "final": 0,
        "missing": []
    }
    model_mode = "gru_60"
    
    try:
        if suffix in [".csv", ".parquet"]:
            logger.info(f"[FILE] Loading dataframe from {original_filename}")
            df = pd.read_parquet(file_path) if suffix == ".parquet" else pd.read_csv(file_path, low_memory=False)

            df.columns = df.columns.str.strip().str.lower()
            feat_cols_lower = [c.strip().lower() for c in _feat_cols]
            orig_cols_lower = [c.strip().lower() for c in ORIGINAL_FEATURES]
            part_cols_lower = [c.strip().lower() for c in PARTIAL_MODEL_FEATURES]

            for col in list(df.columns):
                if col in ['label', 'attack', 'attack_type', 'class', 'label_multiclass', 'label_binary']:
                    df.drop(columns=[col], inplace=True)
            
            # Map columns fuzzily
            mapped_df = pd.DataFrame(index=df.index)
            avail_count = 0
            for c, c_lower in zip(_feat_cols, feat_cols_lower):
                c_clean = ''.join(e for e in c_lower if e.isalnum())
                found = False
                for df_c in df.columns:
                    if ''.join(e for e in df_c if e.isalnum()) == c_clean:
                        mapped_df[c] = df[df_c]
                        avail_count += 1
                        found = True
                        break
                if not found:
                    pass

            has_all_60 = (avail_count == 60)
            
            has_all_original = True
            for c, c_lower in zip(ORIGINAL_FEATURES, orig_cols_lower):
                if c not in mapped_df.columns:
                    has_all_original = False
                    break
            
            has_all_partial = True
            for c, c_lower in zip(PARTIAL_MODEL_FEATURES, part_cols_lower):
                if c not in mapped_df.columns:
                    has_all_partial = False
                    break

            if has_all_60:
                logger.info("[FILE] All 60 features found.")
                df = mapped_df[_feat_cols].fillna(0.0)
                model_mode = "gru_60"
                feature_coverage["available"] = 60
                feature_coverage["final"] = 60
                feature_coverage["engineered"] = 0
            elif has_all_original:
                logger.info("[FILE] 46 original features found. Running feature engineering...")
                df_eng = engineer_features(mapped_df)
                df = df_eng[_feat_cols].fillna(0.0)
                model_mode = "gru_60"
                feature_coverage["available"] = 46
                feature_coverage["final"] = 60
                feature_coverage["engineered"] = 14
            elif has_all_partial:
                logger.warning("[FILE] Missing features. Falling back to Partial PyTorch Model.")
                if _partial_model is None:
                    raise ValueError("Partial model not loaded but required for this dataset.")
                df = mapped_df[PARTIAL_MODEL_FEATURES].fillna(0.0)
                model_mode = "partial_pytorch"
                feature_coverage["required"] = 60
                feature_coverage["available"] = len(PARTIAL_MODEL_FEATURES)
                feature_coverage["final"] = len(PARTIAL_MODEL_FEATURES)
                missing = [c for c in _feat_cols if c not in df.columns]
                feature_coverage["missing"] = missing
            else:
                raise ValueError("Insufficient features for any model mode.")

            n_flows = len(df)
            duration_s = 0.0
            n_zeek_flows = n_flows
            zeek_stats = {}
            xgb_summary = None

        else:
            logger.info(f"[PCAP] Running Zeek on {original_filename}")
            work_dir, zeek_stats = run_zeek(file_path)
            conn_df = parse_conn_log(work_dir / "conn.log")
            n_zeek_flows = len(conn_df)
            
            df = _feat_extractor.extract(file_path)
            n_flows = len(df)
            
            # Use exact existing logic
            X_raw = df.values.astype(np.float32)
            model_mode = "gru_60"
            feature_coverage["available"] = 60
            feature_coverage["final"] = 60
            
            duration_s = 0.0
            xgb_summary = None
            if _xgb_binary is not None:
                xgb_probs  = _xgb_binary.predict_proba(X_raw)[:, 1]
                xgb_preds  = (xgb_probs >= 0.5).astype(int)
                n_attack   = int(xgb_preds.sum())
                xgb_summary = {
                    "total_flows": n_flows,
                    "attack_flows": n_attack,
                    "benign_flows": n_flows - n_attack,
                    "attack_percentage": round(100 * n_attack / n_flows, 1),
                    "mean_attack_prob": float(xgb_probs.mean()),
                }

        # ── RUN INFERENCE ──
        if model_mode == "gru_60":
            logger.info("[FILE] Running 60-feature GRU Pipeline")
            X_raw = df.values.astype(np.float32)
            X_scaled = _scaler.transform(X_raw).astype(np.float32)
            
            if n_flows < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - n_flows, X_scaled.shape[1]), dtype=np.float32)
                X_padded = np.vstack([pad, X_scaled])
            else:
                X_padded = X_scaled

            last_seq = X_padded[-SEQ_LEN:]
            attack_probs, class_probs, _ = _rollout(last_seq, K=ROLLOUT_K)
            rs = _risk_score(attack_probs, class_probs)
            rl = _risk_level(rs)
            mitre = _build_mitre_progression(class_probs)
            sal = _gradient_saliency(last_seq, top_k=10)
            
            forecast = [
                {
                    "step": m["step"],
                    "attack_prob": round(float(attack_probs[m["step"] - 1]), 4),
                    "risk_score": round(_risk_score(attack_probs[:m["step"]], class_probs[:m["step"]]), 2),
                } for m in mitre
            ]

        else:
            logger.info("[FILE] Running Partial PyTorch Pipeline")
            X_raw = df.values.astype(np.float32)
            X_scaled = _partial_scaler.transform(X_raw).astype(np.float32)
            
            # Partial model takes aggregate of flow or just the last flow. 
            # We will use the mean of the scaled features to represent the network state
            mean_state = torch.tensor(X_scaled.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(_device)
            
            with torch.no_grad():
                out = _partial_model(mean_state)
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                
            attack_prob = float(1.0 - probs[0]) # index 0 is BENIGN
            
            # Heuristic risk for partial model
            rs = min(round(attack_prob * 100 * 1.2, 1), 100.0)
            rl = _risk_level(rs)
            
            class_probs = [probs.tolist()]
            mitre = _build_mitre_progression(class_probs)
            sal = []
            forecast = None

        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return {
            "status": "success",
            "model_mode": model_mode,
            "feature_coverage": feature_coverage,
            "input": {
                "filename": original_filename,
                "type": suffix.strip("."),
                "size_bytes": file_path.stat().st_size,
            },
            "summary": {
                "flows_analyzed": n_flows,
                "zeek_flows": n_zeek_flows,
                "duration_seconds": duration_s,
            },
            "xgb_flow_analysis": xgb_summary,
            "current_state": {
                "risk_score": rs,
                "risk_level": rl,
            },
            "attack_probability": float(attack_probs[0]) if model_mode == "gru_60" else attack_prob,
            "forecast": forecast,
            "mitre_progression": mitre,
            "top_saliency_features": sal,
            "processing_time_ms": elapsed_ms,
        }
    finally:
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)"""

content = content[:start_idx] + new_func + content[end_idx:]
file_path.write_text(content)
print("Rewritten successfully.")
