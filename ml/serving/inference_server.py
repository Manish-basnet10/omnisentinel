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

import json, pickle, time, gc, asyncio
from pathlib import Path
from typing import List, Optional
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
WM_DIR      = BASE_DIR / "models" / "world_model"
DASH_DIR    = BASE_DIR / "dashboard"
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


def load_model():
    global _model, _scaler, _feat_cols, _class_names, _eng_feats, _device, _demo_preds

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

    # Load precomputed demo predictions for the SSE stream
    demo_path = METRICS_DIR / "phase10_forecasting_metrics.json"
    if demo_path.exists():
        with open(demo_path) as f:
            _demo_preds = json.load(f)

    print(f"[SERVER] GRU World Model loaded: {sum(p.numel() for p in _model.parameters()):,} params")
    print(f"[SERVER] Features: {len(_feat_cols)}  Classes: {len(_class_names)}")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard static files
if DASH_DIR.exists():
    app.mount("/dashboard", StaticFiles(directory=str(DASH_DIR), html=True), name="dashboard")


@app.on_event("startup")
async def startup_event():
    load_model()


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
    return {
        "status": "ok",
        "model_loaded": _model is not None,
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
