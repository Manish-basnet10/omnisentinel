import re
from pathlib import Path

file_path = Path("ml/serving/inference_server.py")
content = file_path.read_text()

# We want to replace _analyze_pcap_sync and analyze_pcap with our new logic.
# Also we need to import pandas.
if "import pandas as pd" not in content:
    content = content.replace("import numpy as np", "import numpy as np\nimport pandas as pd")

# Let's find the start of _analyze_pcap_sync
start_idx = content.find("def _analyze_pcap_sync(pcap_path: Path, original_filename: str) -> dict:")
if start_idx == -1:
    print("Could not find _analyze_pcap_sync")
    exit(1)

# Let's find the end of analyze_pcap
end_idx = content.find("        logger.info(\"[PCAP] Upload temp dir cleaned up\")")
if end_idx == -1:
    print("Could not find end of analyze_pcap")
    exit(1)
end_idx = content.find("\n", end_idx + 60) # advance past the line

new_funcs = """def _analyze_file_sync(file_path: Path, original_filename: str) -> dict:
    \"\"\"
    Synchronous analysis pipeline — handles PCAP, PCAPNG, CSV, Parquet.
    \"\"\"
    t_start = time.perf_counter()
    work_dir: Optional[Path] = None
    suffix = file_path.suffix.lower()

    try:
        if suffix in [".csv", ".parquet"]:
            logger.info(f"[FILE] Loading dataframe from {original_filename}")
            if suffix == ".parquet":
                df = pd.read_parquet(file_path)
            else:
                df = pd.read_csv(file_path, low_memory=False)

            # Strip whitespace and lowercase all columns
            df.columns = df.columns.str.strip().str.lower()
            feat_cols_lower = [c.strip().lower() for c in _feat_cols]

            # Drop label columns
            for col in list(df.columns):
                if col in ['label', 'attack', 'attack_type', 'class']:
                    df.drop(columns=[col], inplace=True)

            mapped_cols = {}
            missing = []
            for c, c_lower in zip(_feat_cols, feat_cols_lower):
                if c_lower in df.columns:
                    mapped_cols[c] = df[c_lower]
                else:
                    c_clean = ''.join(e for e in c_lower if e.isalnum())
                    found = False
                    for df_c in df.columns:
                        if ''.join(e for e in df_c if e.isalnum()) == c_clean:
                            mapped_cols[c] = df[df_c]
                            found = True
                            break
                    if not found:
                        missing.append(c)

            if missing:
                raise ValueError(f"File is missing {len(missing)} required features: {missing[:5]}...")

            df = pd.DataFrame(mapped_cols)
            df = df[_feat_cols]
            df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            n_flows = len(df)
            logger.info(f"[FILE] Loaded {n_flows} rows from dataframe.")
            duration_s = 0.0
            schema_report = {"schema": "MATCH", "expected": len(_feat_cols), "generated": len(_feat_cols), "missing": [], "extra": [], "order_match": True}
            xgb_summary = None
            n_zeek_flows = n_flows
            zeek_stats = {}
            
        else:
            # ── STEP 1: Zeek ─────────────────────────────────────────────────────
            logger.info(f"[PCAP] Running Zeek on {original_filename}")
            work_dir, zeek_stats = run_zeek(file_path)

            conn_df = parse_conn_log(work_dir / "conn.log")
            n_zeek_flows = len(conn_df)
            logger.info(f"[PCAP] Zeek: {n_zeek_flows:,} flows in conn.log")

            # ── STEP 2: Feature extraction (scapy) ───────────────────────────────
            logger.info("[PCAP] Extracting CICFlowMeter features (scapy)...")
            if _feat_extractor is None:
                raise RuntimeError("PCAPFeatureExtractor is not available (scapy not installed).")
            df = _feat_extractor.extract(file_path)
            n_flows = len(df)
            logger.info(f"[PCAP] Feature extraction complete: {n_flows:,} flows × {len(df.columns)} features")

            # ── STEP 3: Feature schema validation ────────────────────────────────
            schema_report = _feat_extractor.validate_schema(df)
            
            # PCAP duration
            duration_s = 0.0
            if "ts" in conn_df.columns:
                try:
                    ts_vals = pd.to_numeric(conn_df["ts"], errors="coerce").dropna()
                    if len(ts_vals) > 0:
                        duration_s = round(float(ts_vals.max() - ts_vals.min()), 2)
                except Exception:
                    pass
            xgb_summary = None # computed below

        # ── STEP 4: Preprocessing (existing scaler — transform only) ─────────
        logger.info("[FILE] Applying existing StandardScaler (transform only)...")
        X_raw    = df.values.astype(np.float32)
        X_scaled = _scaler.transform(X_raw).astype(np.float32)

        # ── STEP 5: XGBoost per-flow binary classification ───────────────────
        if _xgb_binary is not None and suffix in [".pcap", ".pcapng"]:
            logger.info("[PCAP] Running XGBoost binary classifier...")
            xgb_probs  = _xgb_binary.predict_proba(X_raw)[:, 1]
            xgb_preds  = (xgb_probs >= 0.5).astype(int)
            n_attack   = int(xgb_preds.sum())
            n_benign   = n_flows - n_attack
            mean_attack_prob = float(xgb_probs.mean())
            xgb_summary = {
                "total_flows":        n_flows,
                "attack_flows":       n_attack,
                "benign_flows":       n_benign,
                "attack_percentage":  round(100 * n_attack / n_flows, 1),
                "mean_attack_prob":   round(mean_attack_prob, 4),
            }

        # ── STEP 6: GRU sequence construction + K-step rollout ───────────────
        logger.info("[FILE] Constructing GRU sequences...")
        if n_flows < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - n_flows, X_scaled.shape[1]), dtype=np.float32)
            X_padded = np.vstack([pad, X_scaled])
            logger.warning(f"[FILE] Only {n_flows} flows — padded to SEQ_LEN={SEQ_LEN} with zeros.")
        else:
            X_padded = X_scaled

        last_seq = X_padded[-SEQ_LEN:]

        logger.info("[FILE] Running GRU K-step rollout...")
        attack_probs, class_probs, next_states = _rollout(last_seq, K=ROLLOUT_K)

        # ── STEP 7: Risk + MITRE + saliency ──────────────────────────────────
        rs    = _risk_score(attack_probs, class_probs)
        rl    = _risk_level(rs)
        mitre = _build_mitre_progression(class_probs)
        sal   = _gradient_saliency(last_seq, top_k=10)

        n_windows = max(1, n_flows - SEQ_LEN + 1)
        elapsed_ms = round((time.perf_counter() - t_start) * 1000, 1)

        return {
            "status": "success",
            "input": {
                "filename":    original_filename,
                "type":        suffix.strip("."),
                "size_bytes":  file_path.stat().st_size,
            },
            "summary": {
                "flows_analyzed":     n_flows,
                "zeek_flows":         n_zeek_flows,
                "time_windows":       n_windows,
                "duration_seconds":   duration_s,
                "zeek_version":       zeek_stats.get("zeek_version"),
                "zeek_elapsed_s":     zeek_stats.get("elapsed_seconds"),
            },
            "feature_validation": schema_report,
            "xgb_flow_analysis":  xgb_summary,
            "current_state": {
                "risk_score":  rs,
                "risk_level":  rl,
            },
            "forecast": [
                {
                    "step":        m["step"],
                    "attack_prob": round(float(attack_probs[m["step"] - 1]), 4),
                    "risk_score":  round(_risk_score(attack_probs[:m["step"]], class_probs[:m["step"]]), 2),
                }
                for m in mitre
            ],
            "mitre_progression":      mitre,
            "top_saliency_features":  sal,
            "processing_time_ms":     elapsed_ms,
        }

    finally:
        if work_dir and work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/api/analyze-pcap")
async def analyze_pcap(file: UploadFile = File(...), current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if _model is None:
        raise HTTPException(status_code=503, detail="GRU model not loaded. Server is starting up.")

    original_filename = file.filename or "upload.file"
    logger.info(f"[FILE] Upload received: {original_filename}")

    suffix  = Path(original_filename).suffix.lower()
    if suffix not in [".pcap", ".pcapng", ".csv", ".parquet"]:
        raise HTTPException(status_code=400, detail="Unsupported file extension.")
        
    tmp_dir = Path(tempfile.mkdtemp(prefix="omnisentinel_upload_"))
    file_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        written = 0
        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    out.close()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum allowed: {MAX_FILE_SIZE // 1024 // 1024} MB.",
                    )
                out.write(chunk)

        try:
            if suffix in [".pcap", ".pcapng"]:
                validate_pcap(file_path)
        except PCAPValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid PCAP: {exc}")

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _analyze_file_sync,
                file_path,
                original_filename,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Feature extraction failed: {exc}")
        except Exception as exc:
            logger.exception(f"[FILE] Unexpected error during analysis")
            raise HTTPException(status_code=500, detail=str(exc))

        if db is not None:
            forecast_doc = {
                "_id": uuid.uuid4().hex,
                "user_id": current_user["id"],
                "timestamp": datetime.utcnow(),
                "source_type": suffix.strip("."),
                "filename": original_filename,
                "current_stage": result["current_state"]["risk_level"],
                "current_risk": result["current_state"]["risk_score"],
                "predicted_stage": result["mitre_progression"][0]["tactic"] if result["mitre_progression"] else None,
                "predicted_probability": result["mitre_progression"][0]["probability"] if result["mitre_progression"] else None,
                "forecast_horizon": 8,
                "forecast_steps": result["forecast"],
                "mitre_tactics": [m["tactic"] for m in result["mitre_progression"]],
                "mitre_techniques": [m["technique"] for m in result["mitre_progression"]],
                "explanation": result["top_saliency_features"],
                "model_name": "GRU World Model"
            }
            await db["forecasts"].insert_one(forecast_doc)
            
            if result["current_state"]["risk_level"] in ["HIGH", "CRITICAL"]:
                alert_doc = {
                    "_id": uuid.uuid4().hex,
                    "user_id": current_user["id"],
                    "timestamp": datetime.utcnow(),
                    "severity": 4 if result["current_state"]["risk_level"] == "HIGH" else 5,
                    "current_stage": result["current_state"]["risk_level"],
                    "predicted_stage": forecast_doc["predicted_stage"],
                    "risk_score": forecast_doc["current_risk"],
                    "probability": forecast_doc["predicted_probability"],
                    "mitre_tactic": forecast_doc["predicted_stage"],
                    "status": "new"
                }
                await db["alerts"].insert_one(alert_doc)
                
            state_doc = {
                "_id": uuid.uuid4().hex,
                "user_id": current_user["id"],
                "timestamp": datetime.utcnow(),
                "current_stage": result["current_state"]["risk_level"],
                "risk_score": result["current_state"]["risk_score"],
                "source": suffix.strip("."),
                "forecast_id": forecast_doc["_id"]
            }
            await db["network_states"].insert_one(state_doc)

        return JSONResponse(content=result)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
"""

content = content[:start_idx] + new_funcs + content[end_idx:]

# Additionally, the pcap validator might throw if it's not a PCAP, but we conditionally call it now.
file_path.write_text(content)
print("Updated inference_server.py successfully")
