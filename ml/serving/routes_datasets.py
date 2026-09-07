"""
routes_datasets.py — Dataset Session Management
================================================
Provides the dataset-centric API that powers the global frontend state.

POST   /api/datasets/upload        — Upload file, run ML pipeline once, store result
GET    /api/datasets               — List user's datasets (metadata only)
GET    /api/datasets/{id}          — Dataset metadata
GET    /api/datasets/{id}/analysis — Full canonical analysis (all frontend pages read this)
DELETE /api/datasets/{id}          — Remove dataset + session

Reuses the existing _analyze_file_sync pipeline — no duplicate ML logic.
"""

import uuid, shutil, tempfile, asyncio, logging
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from ml.serving.auth import get_current_user
from ml.serving.database import get_db

logger = logging.getLogger("omnisentinel.datasets")

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# ── Injected at server startup (avoids circular imports) ──────────────────────
_analyze_file_sync_fn = None   # set by inference_server.py
_model_ready_fn       = None   # set by inference_server.py
MAX_FILE_SIZE = 200 * 1024 * 1024   # 200 MB


def register_pipeline(analyze_fn, model_ready_fn):
    """Called from inference_server.py after model loads."""
    global _analyze_file_sync_fn, _model_ready_fn
    _analyze_file_sync_fn = analyze_fn
    _model_ready_fn       = model_ready_fn


# ══════════════════════════════════════════════════════════════════════════════
# POST /api/datasets/upload
# ══════════════════════════════════════════════════════════════════════════════
@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    db = Depends(get_db),
):
    if _analyze_file_sync_fn is None or (_model_ready_fn and not _model_ready_fn()):
        raise HTTPException(status_code=503, detail="ML pipeline not ready.")

    original_filename = file.filename or "upload.file"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in [".pcap", ".pcapng", ".csv", ".parquet"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .parquet, .pcap, or .pcapng.")

    dataset_id = uuid.uuid4().hex
    tmp_dir    = Path(tempfile.mkdtemp(prefix="omnisentinel_ds_"))
    file_path  = tmp_dir / f"{dataset_id}{suffix}"

    try:
        # Stream file to disk
        written = 0
        with open(file_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"File too large. Max: {MAX_FILE_SIZE // 1024 // 1024} MB")
                out.write(chunk)

        logger.info(f"[DATASET] Received '{original_filename}' ({written / 1024:.1f} KB) → dataset_id={dataset_id}")

        # ── Run existing ML pipeline (runs in thread, non-blocking) ───────────
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                _analyze_file_sync_fn,
                file_path,
                original_filename,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Feature extraction failed: {exc}")
        except Exception as exc:
            logger.exception("[DATASET] Analysis pipeline error")
            raise HTTPException(status_code=500, detail=str(exc))

        # ── Build canonical analysis session ──────────────────────────────────
        now = datetime.utcnow()
        fc  = result.get("feature_coverage", {})
        mp  = result.get("mitre_progression", [])

        # Risk trend from forecast (compare first step to current)
        forecast_steps = result.get("forecast") or []
        current_risk   = result["current_state"]["risk_score"]
        if forecast_steps and isinstance(forecast_steps, list) and len(forecast_steps) > 0:
            last_step_risk = forecast_steps[-1].get("risk_score", current_risk)
            risk_trend = "increasing" if last_step_risk > current_risk else "decreasing" if last_step_risk < current_risk else "stable"
        else:
            risk_trend = "unknown"

        prediction = {
            "current_risk":          round(current_risk, 2),
            "risk_level":            result["current_state"]["risk_level"],
            "risk_trend":            risk_trend,
            "attack_probability":    round(float(result.get("attack_probability", 0.0)), 4),
            "predicted_next_stage":  mp[0]["tactic"] if mp else None,
            "predicted_technique":   mp[0]["technique"] if mp else None,
            "confidence":            round(float(mp[0]["probability"]) * 100, 1) if mp else None,
        }

        # Build MITRE stages with status markers
        mitre_data = []
        for i, m in enumerate(mp):
            mitre_data.append({
                **m,
                "status": "current" if i == 0 else "forecast",
            })

        # Build timeline events from mitre progression
        timeline = []
        for i, m in enumerate(mp):
            timeline.append({
                "id":       i + 1,
                "label":    m["tactic"],
                "stage":    m["top_class"],
                "detail":   f"MITRE {m['technique']} — probability {round(m['probability'] * 100, 1)}%",
                "observed": i == 0,
                "riskScore": forecast_steps[i].get("risk_score", current_risk) if i < len(forecast_steps) else current_risk,
            })

        # Generate alerts based on risk
        risk_level = result["current_state"]["risk_level"]
        alerts = []
        if risk_level in ("HIGH", "CRITICAL"):
            alerts.append({
                "id":       uuid.uuid4().hex,
                "severity": "Critical" if risk_level == "CRITICAL" else "High",
                "title":    f"{'Critical' if risk_level == 'CRITICAL' else 'High'} Risk Detected — {original_filename}",
                "detail":   f"Risk score {current_risk:.1f}. Predicted stage: {prediction['predicted_next_stage'] or 'Unknown'}",
                "stage":    mp[0]["top_class"] if mp else "Unknown",
                "confidence": prediction["confidence"] or 0,
                "time":     now.strftime("%H:%M:%S"),
                "status":   "New",
                "source":   f"ML Analysis ({result.get('model_mode','gru_60')})",
                "dataset_id": dataset_id,
            })
        if prediction["attack_probability"] and prediction["attack_probability"] > 0.7:
            alerts.append({
                "id":       uuid.uuid4().hex,
                "severity": "High",
                "title":    f"High Attack Probability — {round(prediction['attack_probability']*100,1)}%",
                "detail":   f"Model predicts attack with {round(prediction['attack_probability']*100,1)}% confidence",
                "stage":    mp[0]["top_class"] if mp else "Unknown",
                "confidence": round(prediction["attack_probability"] * 100, 1),
                "time":     now.strftime("%H:%M:%S"),
                "status":   "New",
                "source":   f"ML Analysis ({result.get('model_mode','gru_60')})",
                "dataset_id": dataset_id,
            })

        # Investigations from top observed features + mitre context
        top_features = [s["feature"] for s in (result.get("top_saliency_features") or [])[:5]]
        investigations = []
        if mp:
            investigations.append({
                "id":           f"INV-{dataset_id[:6].upper()}",
                "title":        f"Anomalous Traffic — {original_filename}",
                "severity":     risk_level,
                "status":       "Open",
                "observations": [
                    f"Risk score: {current_risk:.1f} ({risk_level})",
                    f"Attack probability: {round(prediction['attack_probability']*100,1)}%",
                    f"Top indicators: {', '.join(top_features) if top_features else 'N/A'}",
                ],
                "predictedBehavior": [
                    f"Predicted stage: {m['tactic']} ({m['technique']})" for m in mp[:3]
                ],
                "recommendedSteps": [
                    f"Investigate traffic patterns related to {mp[0]['tactic']} if applicable",
                    "Review flagged features for anomalies",
                    "Escalate if risk remains above threshold in next observation window",
                    "Cross-reference with MITRE ATT&CK playbook",
                ],
                "affectedAssets": ["Network Segment (inferred from dataset)"],
                "evidence": [
                    {
                        "id":           1,
                        "time":         now.strftime("%H:%M:%S"),
                        "type":         "ML Analysis",
                        "description":  f"PyTorch model ({result.get('model_mode','gru_60')}) analysed {result['summary']['flows_analyzed']} flows from {original_filename}",
                    }
                ],
                "dataset_id": dataset_id,
            })

        # Dataset metadata doc
        dataset_doc = {
            "_id":          dataset_id,
            "user_id":      current_user["id"],
            "filename":     original_filename,
            "format":       suffix.strip("."),
            "row_count":    result["summary"]["flows_analyzed"],
            "feature_count": fc.get("available", 0),
            "model_mode":   result.get("model_mode", "gru_60"),
            "created_at":   now,
        }

        # Analysis session doc
        session_doc = {
            "_id":        dataset_id,   # same as dataset_id for easy lookup
            "dataset_id": dataset_id,
            "user_id":    current_user["id"],
            "created_at": now,
            "model": {
                "type":         "GRU World Model" if "gru" in result.get("model_mode","") else "Partial PyTorch MLP",
                "mode":         result.get("model_mode", "gru_60"),
                "features_used": fc.get("final", 0),
            },
            "dataset": {
                "rows":               result["summary"]["flows_analyzed"],
                "features_available": fc.get("available", 0),
                "features_used":      fc.get("final", 0),
                "features_engineered": fc.get("engineered", 0),
                "missing_features":   fc.get("missing", []),
                "format":             suffix.strip("."),
                "filename":           original_filename,
            },
            "prediction":         prediction,
            "forecast":           forecast_steps,
            "mitre_progression":  mitre_data,
            "timeline":           timeline,
            "alerts":             alerts,
            "investigations":     investigations,
            "top_saliency_features": result.get("top_saliency_features", []),
            "traffic_analysis": {
                "flows_analyzed":    result["summary"]["flows_analyzed"],
                "duration_seconds":  result["summary"].get("duration_seconds", 0),
                "feature_coverage":  fc,
                "xgb_flow_analysis": result.get("xgb_flow_analysis"),
                "processing_time_ms": result.get("processing_time_ms", 0),
            },
            "model_metrics": {
                "model_mode":         result.get("model_mode", "gru_60"),
                "features_required":  fc.get("required", 60),
                "features_available": fc.get("available", 0),
                "features_engineered": fc.get("engineered", 0),
                "features_final":     fc.get("final", 0),
                "processing_time_ms": result.get("processing_time_ms", 0),
            },
        }

        # ── Persist to MongoDB ────────────────────────────────────────────────
        if db is not None:
            await db["datasets"].insert_one(dataset_doc)
            await db["analysis_sessions"].insert_one(session_doc)

            # Also persist alerts individually for the alerts endpoint
            for alert in alerts:
                await db["alerts"].insert_one({
                    "_id":          alert["id"],
                    "user_id":      current_user["id"],
                    "dataset_id":   dataset_id,
                    "timestamp":    now,
                    "severity":     5 if alert["severity"] == "Critical" else 4,
                    "current_stage": alert["stage"],
                    "predicted_stage": prediction["predicted_next_stage"],
                    "risk_score":   current_risk,
                    "probability":  prediction["attack_probability"],
                    "mitre_tactic": prediction["predicted_next_stage"],
                    "status":       "new",
                    "title":        alert["title"],
                    "detail":       alert["detail"],
                })

            # Legacy: also write to forecasts + network_states for backwards compat
            forecast_doc = {
                "_id":                uuid.uuid4().hex,
                "user_id":            current_user["id"],
                "dataset_id":         dataset_id,
                "timestamp":          now,
                "source_type":        suffix.strip("."),
                "filename":           original_filename,
                "current_stage":      result["current_state"]["risk_level"],
                "current_risk":       current_risk,
                "predicted_stage":    prediction["predicted_next_stage"],
                "predicted_probability": prediction["confidence"],
                "forecast_horizon":   8,
                "forecast_steps":     forecast_steps,
                "mitre_tactics":      [m["tactic"] for m in mp],
                "mitre_techniques":   [m["technique"] for m in mp],
                "explanation":        result.get("top_saliency_features", []),
                "model_name":         session_doc["model"]["type"],
            }
            await db["forecasts"].insert_one(forecast_doc)

        logger.info(f"[DATASET] Analysis stored. dataset_id={dataset_id} risk={current_risk:.1f} level={risk_level}")

        return JSONResponse({
            "dataset_id": dataset_id,
            "filename":   original_filename,
            "status":     "complete",
            "summary": {
                "rows":        result["summary"]["flows_analyzed"],
                "model_mode":  result.get("model_mode", "gru_60"),
                "risk_level":  risk_level,
                "risk_score":  round(current_risk, 2),
                "features_available": fc.get("available", 0),
                "features_final":     fc.get("final", 0),
            },
        })

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/datasets
# ══════════════════════════════════════════════════════════════════════════════
@router.get("")
async def list_datasets(current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        return []
    docs = await db["datasets"].find({"user_id": current_user["id"]}).sort("created_at", -1).to_list(50)
    for d in docs:
        d["created_at"] = d["created_at"].isoformat() if hasattr(d.get("created_at"), "isoformat") else str(d.get("created_at", ""))
    return docs


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/datasets/{id}
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    doc = await db["datasets"].find_one({"_id": dataset_id, "user_id": current_user["id"]})
    if not doc:
        raise HTTPException(status_code=404, detail="Dataset not found")
    doc["created_at"] = doc["created_at"].isoformat() if hasattr(doc.get("created_at"), "isoformat") else str(doc.get("created_at", ""))
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# GET /api/datasets/{id}/analysis — CANONICAL RESULT used by all frontend pages
# ══════════════════════════════════════════════════════════════════════════════
@router.get("/{dataset_id}/analysis")
async def get_analysis(dataset_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    doc = await db["analysis_sessions"].find_one({"dataset_id": dataset_id, "user_id": current_user["id"]})
    if not doc:
        raise HTTPException(status_code=404, detail="Analysis session not found")
    doc["created_at"] = doc["created_at"].isoformat() if hasattr(doc.get("created_at"), "isoformat") else str(doc.get("created_at", ""))
    return doc


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /api/datasets/{id}
# ══════════════════════════════════════════════════════════════════════════════
@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, current_user: dict = Depends(get_current_user), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    r = await db["datasets"].delete_one({"_id": dataset_id, "user_id": current_user["id"]})
    if r.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Dataset not found")
    await db["analysis_sessions"].delete_one({"dataset_id": dataset_id})
    await db["alerts"].delete_many({"dataset_id": dataset_id})
    return {"message": "Dataset deleted"}
