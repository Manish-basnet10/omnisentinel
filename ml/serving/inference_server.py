"""
Phase 11 – Inference Server
============================
FastAPI server exposing the GRU World Model for real-time
attack forecasting with K-step rollout, risk scoring,
MITRE ATT&CK mapping, and gradient saliency explainability.

Endpoints:
  GET  /health          — liveness check
  GET  /model-info      — model metadata
  POST /predict         — single sequence forecast (rollout K steps)
  POST /batch-predict   — batch of sequences
  GET  /demo-stream     — SSE: streams precomputed test predictions live

Run:
  uvicorn ml.serving.inference_server:app --host 0.0.0.0 --port 8000 --reload
"""

import json, pickle, time, gc, asyncio, logging, shutil, tempfile, uuid
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from ml.serving.database import connect_to_mongo, close_mongo_connection, db_instance, get_db
from ml.serving.auth import get_current_user
from ml.serving.routes_auth import router as auth_router
from ml.serving.routes_api import router as api_router
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── PCAP pipeline services ────────────────────────────────────────────────────
from ml.serving.services.pcap_validator import validate_pcap, PCAPValidationError, MAX_FILE_SIZE
from ml.serving.services.zeek_service  import (
    run_zeek, parse_conn_log, get_zeek_version, find_zeek,
    ZeekNotFoundError, ZeekExecutionError,
)
from ml.serving.services.feature_service import PCAPFeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("omnisentinel.server")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
WM_DIR      = BASE_DIR / "models" / "world_model"

METRICS_DIR = BASE_DIR / "results" / "metrics"

# ── Constants ─────────────────────────────────────────────────────────────────
SEQ_LEN      = 10
ROLLOUT_K    = 8
N_CLASSES_MC = 15
RISK_DISCOUNT = 0.85
RISK_THRESH   = {"low": 20, "medium": 40, "high": 65, "critical": 85}

MITRE_MAP = {
    0:  {"label": "BENIGN",               "tactic": None,                "stage": 0, "severity": 0, "technique": "—"},
    1:  {"label": "Bot",                  "tactic": "Command & Control",  "stage": 6, "severity": 5, "technique": "T1071"},
    2:  {"label": "DDoS",                 "tactic": "Impact",             "stage": 7, "severity": 5, "technique": "T1498"},
    3:  {"label": "DoS_GoldenEye",        "tactic": "Impact",             "stage": 7, "severity": 4, "technique": "T1499"},
    4:  {"label": "DoS_Hulk",             "tactic": "Impact",             "stage": 7, "severity": 4, "technique": "T1499"},
    5:  {"label": "DoS_Slowhttptest",     "tactic": "Impact",             "stage": 7, "severity": 3, "technique": "T1499.001"},
    6:  {"label": "DoS_slowloris",        "tactic": "Impact",             "stage": 7, "severity": 3, "technique": "T1499.001"},
    7:  {"label": "FTP-Patator",          "tactic": "Credential Access",  "stage": 3, "severity": 3, "technique": "T1110.003"},
    8:  {"label": "Heartbleed",           "tactic": "Initial Access",     "stage": 2, "severity": 5, "technique": "T1190"},
    9:  {"label": "Infiltration",         "tactic": "Lateral Movement",   "stage": 5, "severity": 5, "technique": "T1210"},
    10: {"label": "PortScan",             "tactic": "Discovery",          "stage": 1, "severity": 2, "technique": "T1046"},
    11: {"label": "SSH-Patator",          "tactic": "Credential Access",  "stage": 3, "severity": 3, "technique": "T1110.003"},
    12: {"label": "WebAttack_BruteForce", "tactic": "Credential Access",  "stage": 3, "severity": 3, "technique": "T1110"},
    13: {"label": "WebAttack_SQL_Inj",    "tactic": "Initial Access",     "stage": 2, "severity": 5, "technique": "T1190"},
    14: {"label": "WebAttack_XSS",        "tactic": "Execution",          "stage": 4, "severity": 3, "technique": "T1059.007"},
}

SMAP = {i: MITRE_MAP[i]["severity"] for i in range(N_CLASSES_MC)}


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════════
class GRUWorldModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, n_classes_mc, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.head_state  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(hidden_dim, input_dim))
        self.head_binary = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, 2))
        self.head_mc     = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(),
                                          nn.Dropout(dropout), nn.Linear(64, n_classes_mc))

    def forward(self, x):
        out, _ = self.gru(x)
        last   = self.layer_norm(out[:, -1, :])
        return self.head_state(last), self.head_binary(last), self.head_mc(last)


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL MODEL STATE (loaded once at startup)
# ══════════════════════════════════════════════════════════════════════════════
_model      = None
_scaler     = None
_feat_cols  = None
_class_names= None
_eng_feats  = None
_device     = None
_demo_preds = None   # precomputed demo stream predictions
_xgb_binary = None   # XGBoost binary classifier (for PCAP per-flow analysis)
_feat_extractor = None  # PCAPFeatureExtractor (lazy-init)


def load_model():
    global _model, _scaler, _feat_cols, _class_names, _eng_feats, _device, _demo_preds
    global _xgb_binary, _feat_extractor

    device = torch.device("cpu")   # CPU for low-latency inference
    _device = device

    with open(PROC_DIR / "feature_list.json")   as f: feat_def  = json.load(f)
    with open(PROC_DIR / "label_encoding.json") as f: label_enc = json.load(f)
    with open(MODELS_DIR / "standard_scaler.pkl", "rb") as f:
        _scaler = pickle.load(f)["scaler"]

    _feat_cols   = feat_def["all_numeric_features"]
    _class_names = label_enc["classes"]
    _eng_feats   = set(feat_def["engineered_features"])

    ckpt = torch.load(WM_DIR / "gru_world_model.pt", map_location=device)
    hp   = ckpt["hyperparams"]
    _model = GRUWorldModel(hp["input_dim"], hp["hidden_dim"], hp["num_layers"],
                            hp["n_classes_mc"], hp["dropout"]).to(device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    # Load XGBoost binary model (for per-flow PCAP classification)
    xgb_path = MODELS_DIR / "xgb_binary.pkl"
    if xgb_path.exists():
        with open(xgb_path, "rb") as f:
            _xgb_binary = pickle.load(f)
        logger.info(f"[SERVER] XGBoost binary loaded from {xgb_path.name}")
    else:
        logger.warning("[SERVER] xgb_binary.pkl not found — XGBoost analysis disabled")

    # Initialise the PCAP feature extractor (requires scapy)
    try:
        _feat_extractor = PCAPFeatureExtractor(_feat_cols)
        logger.info("[SERVER] PCAPFeatureExtractor ready (scapy available)")
    except RuntimeError as exc:
        logger.warning(f"[SERVER] PCAPFeatureExtractor not available: {exc}")

    # Load precomputed demo predictions for the SSE stream
    demo_path = METRICS_DIR / "phase10_forecasting_metrics.json"
    if demo_path.exists():
        with open(demo_path) as f:
            _demo_preds = json.load(f)

    logger.info(f"[SERVER] GRU World Model loaded: {sum(p.numel() for p in _model.parameters()):,} params")
    logger.info(f"[SERVER] Features: {len(_feat_cols)}  Classes: {len(_class_names)}")


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _rollout(seq_scaled: np.ndarray, K: int = ROLLOUT_K):
    """K-step autoregressive rollout. Input: (seq_len, D) scaled."""
    window = seq_scaled.copy()
    attack_probs, class_probs, next_states = [], [], []
    with torch.no_grad():
        for _ in range(K):
            x  = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(_device)
            ps, pb, pm = _model(x)
            p_bin = F.softmax(pb, dim=1).squeeze(0).cpu().numpy()
            p_mc  = F.softmax(pm, dim=1).squeeze(0).cpu().numpy()
            s_hat = ps.squeeze(0).cpu().numpy()
            attack_probs.append(float(p_bin[1]))
            class_probs.append(p_mc.tolist())
            next_states.append(s_hat.tolist())
            window = np.vstack([window[1:], s_hat[np.newaxis, :]])
    return np.array(attack_probs), np.array(class_probs), next_states


def _risk_score(ap: np.ndarray, cp: np.ndarray) -> float:
    K    = len(ap)
    dmax = sum(RISK_DISCOUNT**k for k in range(K)) * 5.0
    raw  = sum(RISK_DISCOUNT**k * ap[k] * (1 + SMAP.get(int(np.argmax(cp[k])), 0) / 5.0)
               for k in range(K))
    return round(min(raw / (dmax * 2.0 / 5.0) * 100, 100), 2)


def _risk_level(score: float) -> str:
    if score >= RISK_THRESH["critical"]: return "CRITICAL"
    if score >= RISK_THRESH["high"]:     return "HIGH"
    if score >= RISK_THRESH["medium"]:   return "MEDIUM"
    if score >= RISK_THRESH["low"]:      return "LOW"
    return "SAFE"


def _gradient_saliency(seq_scaled: np.ndarray, top_k: int = 10) -> list:
    """Gradient saliency: top-K features driving P(ATTACK)."""
    x_t = torch.tensor(seq_scaled[np.newaxis], dtype=torch.float32, requires_grad=True)
    _model.train()
    _model.zero_grad()
    _, pb, _ = _model(x_t)
    F.softmax(pb, dim=1)[0, 1].backward()
    _model.eval()
    if x_t.grad is None:
        return []
    sal = (x_t.grad.detach().abs() * x_t.detach().abs()).numpy()[0].mean(axis=0)
    top_idx = np.argsort(sal)[-top_k:][::-1]
    return [{"feature": _feat_cols[i], "saliency": round(float(sal[i]), 6),
              "engineered": _feat_cols[i] in _eng_feats} for i in top_idx]


def _build_mitre_progression(class_probs: np.ndarray) -> list:
    """Build MITRE stage progression across rollout steps."""
    result = []
    for k, cp in enumerate(class_probs):
        top_cls   = int(np.argmax(cp))
        info      = MITRE_MAP[top_cls]
        result.append({
            "step":       k + 1,
            "top_class":  info["label"],
            "tactic":     info["tactic"],
            "technique":  info["technique"],
            "stage":      info["stage"],
            "severity":   info["severity"],
            "probability": round(float(cp[top_cls]), 4),
        })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="OmniSentinel – Network Attack Forecasting API",
    description="GRU World Model with K-step rollout, MITRE ATT&CK mapping, and risk scoring.",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.include_router(auth_router)
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.on_event("startup")
async def startup_event():
    print("loading model"); load_model(); print("model loaded")
    print("connecting mongo"); await connect_to_mongo(); print("mongo connected")

@app.on_event("shutdown")
async def shutdown_event():
    await close_mongo_connection()


# ── Pydantic schemas ──────────────────────────────────────────────────────────
class SequenceInput(BaseModel):
    """
    A temporal sequence of exactly seq_len=10 network flow feature vectors.
    Each row = one flow, each col = one of the 60 features (raw, unscaled).
    """
    sequence: List[List[float]]   # shape: (10, 60)
    rollout_k: Optional[int] = ROLLOUT_K
    return_saliency: Optional[bool] = True


class BatchInput(BaseModel):
    sequences: List[List[List[float]]]   # (N, 10, 60)
    rollout_k: Optional[int] = ROLLOUT_K


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    mongo_status = "connected" if db_instance.client is not None else "disconnected"
    return {
        "status": "ok" if mongo_status == "connected" else "degraded",
        "model": "GRU World Model" if _model is not None else "Not Loaded",
        "mongodb": mongo_status,
        "timestamp": time.time(),
    }


@app.get("/model-info")
async def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model": "GRU World Model (Phase 8)",
        "parameters": sum(p.numel() for p in _model.parameters()),
        "input_features": len(_feat_cols),
        "seq_len": SEQ_LEN,
        "rollout_k": ROLLOUT_K,
        "n_classes": N_CLASSES_MC,
        "class_names": _class_names,
        "feature_names": _feat_cols,
        "engineered_features": list(_eng_feats),
        "device": str(_device),
    }


@app.post("/predict")
async def predict(req: SequenceInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    seq = np.array(req.sequence, dtype=np.float32)
    if seq.shape != (SEQ_LEN, len(_feat_cols)):
        raise HTTPException(
            status_code=422,
            detail=f"Expected sequence shape ({SEQ_LEN}, {len(_feat_cols)}), got {seq.shape}"
        )

    t0 = time.perf_counter()

    # Scale input (same scaler as training)
    seq_scaled = _scaler.transform(seq).astype(np.float32)

    # K-step rollout
    K          = min(req.rollout_k or ROLLOUT_K, ROLLOUT_K)
    attack_probs, class_probs, next_states = _rollout(seq_scaled, K=K)

    # Risk scoring
    rs    = _risk_score(attack_probs, class_probs)
    rl    = _risk_level(rs)
    mitre = _build_mitre_progression(class_probs)

    # Gradient saliency (optional, slightly slower)
    saliency = _gradient_saliency(seq_scaled) if req.return_saliency else []

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    return {
        "latency_ms": latency_ms,
        "risk_score":  rs,
        "risk_level":  rl,
        "attack_probability_timeline": [round(p, 4) for p in attack_probs.tolist()],
        "mitre_progression": mitre,
        "top_saliency_features": saliency,
        "rollout_k": K,
        "summary": {
            "max_attack_prob":   round(float(attack_probs.max()), 4),
            "mean_attack_prob":  round(float(attack_probs.mean()), 4),
            "predicted_tactic_k1": mitre[0]["tactic"],
            "predicted_class_k1":  mitre[0]["top_class"],
        }
    }


@app.post("/batch-predict")
async def batch_predict(req: BatchInput):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    results = []
    t0 = time.perf_counter()
    for seq_raw in req.sequences:
        seq        = np.array(seq_raw, dtype=np.float32)
        seq_scaled = _scaler.transform(seq).astype(np.float32)
        K          = req.rollout_k or ROLLOUT_K
        ap, cp, _  = _rollout(seq_scaled, K=K)
        rs  = _risk_score(ap, cp)
        rl  = _risk_level(rs)
        results.append({
            "risk_score": rs, "risk_level": rl,
            "attack_probability_timeline": [round(p, 4) for p in ap.tolist()],
            "predicted_tactic_k1": _build_mitre_progression(cp)[0]["tactic"],
        })
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    return {"count": len(results), "latency_ms": latency_ms, "predictions": results}


@app.get("/demo-stream")
async def demo_stream():
    """
    Server-Sent Events: streams simulated live predictions from
    precomputed test sequence data at 1.5s intervals.
    Dashboard subscribes to this for the live feed.
    """
    import random, math

    # Prebuilt demo scenarios (realistic attack/benign progressions)
    SCENARIOS = [
        {"label": "DDoS Attack Surge",          "pattern": "attack_high"},
        {"label": "Port Scan Reconnaissance",   "pattern": "recon"},
        {"label": "Credential Brute Force",     "pattern": "credential"},
        {"label": "Normal Traffic",             "pattern": "benign"},
        {"label": "Command & Control Beacon",   "pattern": "c2"},
        {"label": "Mixed Traffic",              "pattern": "mixed"},
    ]

    MITRE_PATTERNS = {
        "attack_high":  [0.82, 0.85, 0.88, 0.90, 0.87, 0.89, 0.91, 0.88],
        "recon":        [0.35, 0.42, 0.51, 0.58, 0.64, 0.70, 0.65, 0.60],
        "credential":   [0.55, 0.60, 0.65, 0.70, 0.72, 0.68, 0.71, 0.73],
        "benign":       [0.08, 0.09, 0.07, 0.10, 0.08, 0.09, 0.07, 0.08],
        "c2":           [0.72, 0.75, 0.78, 0.80, 0.77, 0.79, 0.81, 0.80],
        "mixed":        [0.30, 0.45, 0.38, 0.52, 0.41, 0.48, 0.55, 0.50],
    }

    TACTIC_BY_PATTERN = {
        "attack_high": "Impact", "recon": "Discovery",
        "credential":  "Credential Access", "benign": None,
        "c2":          "Command & Control", "mixed": "Impact",
    }

    seq_id   = 0
    tick     = 0

    async def event_generator():
        nonlocal seq_id, tick
        while True:
            scenario = SCENARIOS[seq_id % len(SCENARIOS)]
            pattern  = scenario["pattern"]
            base_ap  = MITRE_PATTERNS[pattern]
            # Add jitter for realism
            ap = [min(1.0, max(0.0, p + random.gauss(0, 0.03))) for p in base_ap]
            risk = _risk_score(np.array(ap),
                               np.zeros((8, N_CLASSES_MC)))   # simplified
            rl   = _risk_level(risk)
            tactic = TACTIC_BY_PATTERN.get(pattern)

            payload = {
                "tick":       tick,
                "seq_id":     seq_id,
                "scenario":   scenario["label"],
                "timestamp":  time.strftime("%H:%M:%S"),
                "attack_probability_timeline": [round(p, 3) for p in ap],
                "risk_score": round(risk, 1),
                "risk_level": rl,
                "predicted_tactic_k1": tactic,
                "max_attack_prob": round(max(ap), 3),
                "alert": rl in ("HIGH", "CRITICAL"),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            seq_id += 1
            tick   += 1
            await asyncio.sleep(1.8)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# PCAP ANALYSIS ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/zeek-status")
async def zeek_status():
    """Check whether Zeek is available and return its version."""
    zeek_bin = find_zeek()
    version  = get_zeek_version()
    if not zeek_bin:
        return {
            "zeek_available": False,
            "zeek_binary": None,
            "zeek_version": None,
            "message": (
                "Zeek not found. Install with: brew install zeek\n"
                "Or set ZEEK_PATH=/path/to/zeek"
            ),
        }
    return {
        "zeek_available": True,
        "zeek_binary": zeek_bin,
        "zeek_version": version,
        "message": "Zeek is ready.",
    }


def _analyze_file_sync(file_path: Path, original_filename: str) -> dict:
    """
    Synchronous analysis pipeline — handles PCAP, PCAPNG, CSV, Parquet.
    """
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

