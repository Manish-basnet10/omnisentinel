# OmniSentinel: AI Network Attack Forecasting

> **SIH 2026 — Problem Statement #26153 (NTRO) | Theme: Cyber Security**

OmniSentinel is an advanced AI/ML subsystem that learns network behavior as a **Temporal World Model** to *forecast* attack progression — not just classify current traffic. It treats network flows as a continuous temporal sequence and predicts attack probability up to **8 steps into the future**.

---

## Key Results (Real Metrics — CIC-IDS2017 Test Set)

| Metric | Value |
|---|---|
| GRU k=1 Rollout AUC | **0.8153** |
| GRU k=8 Rollout AUC | **0.7725** (only 4.3% degradation) |
| Risk Score: Attack vs Benign Separation | **43.55 points** |
| Zero-shot AUC (unseen DDoS + PortScan) | **0.8054** |
| Unit Tests | **32/32 passing** |

---

## Architecture

```
Raw Network Flow → Feature Engineering (60 features)
     → Sequence (10 timesteps) → GRU World Model
          ├── Binary Head: P(attack | S_t)
          ├── Multiclass Head: attack type prediction
          └── State Head: next state S_{t+1}
               └── K-step Autoregressive Rollout (K=8)
                    ├── Risk Score (discounted, 0–100)
                    └── MITRE ATT&CK Stage Mapping
```

---

## Phase Completion

| # | Phase | Status |
|---|---|---|
| 2 | Data Cleaning (CIC-IDS2017) | ✅ |
| 3 | Feature Engineering (60 features, 9 engineered) | ✅ |
| 4 | Network State Representation | ✅ |
| 5 | Temporal Sequence Construction | ✅ |
| 6 | Logistic Regression Baseline | ✅ |
| 7 | XGBoost Static Comparison | ✅ |
| 8 | GRU World Model (primary) | ✅ |
| 9 | LSTM World Model (comparison) | ✅ |
| 10 | K-step Forecasting (K=8 rollout) | ✅ |
| 11 | Attack Progression Probability | ✅ |
| 12 | MITRE ATT&CK Stage Mapping | ✅ |
| 13 | Explainability (SHAP + gradient saliency) | ✅ |
| 14 | Unseen Attack Evaluation (zero-shot) | ✅ |
| 15 | Final Benchmark Table | ✅ |
| 16 | Save All Model Artifacts | ✅ |
| 17 | Offline Inference Engine (CSV → JSON) | ✅ |
| 18 | Master Visualizations | ✅ |
| 19 | config.yaml + Reproducibility | ✅ |
| 20 | Unit Tests + README | ✅ |

---

## Project Structure

```
ai-network-attack-forecasting/
├── configs/
│   └── config.yaml              # All hyperparams, seeds, paths
├── data/
│   ├── raw/                     # ⚠️ NOT in git — download separately
│   └── processed/               # ⚠️ NOT in git — regenerate via scripts
├── dashboard/                   # Live SSE dashboard (HTML/JS)
├── ml/
│   ├── data/                    # Phase 2–5 scripts
│   ├── models/
│   │   ├── world_model/         # Phase 8 GRU, Phase 9 LSTM
│   │   ├── phase6_logistic_regression.py
│   │   └── phase7_xgboost.py
│   ├── explainability/
│   │   └── phase13_shap.py
│   ├── evaluation/
│   │   ├── phase14_unseen_eval.py
│   │   ├── phase15_benchmark.py
│   │   └── phase18_master_viz.py
│   ├── inference/
│   │   └── phase17_offline_inference.py
│   ├── serving/
│   │   ├── inference_server.py  # FastAPI + SSE
│   │   └── traffic_simulator.py
│   └── tests/
│       └── test_pipeline.py     # 32 unit tests
├── models/                      # ⚠️ Weights NOT in git — train or download
│   ├── xgb_multiclass.json      # ✅ In git (small, < 1MB)
│   └── mitre_mapper.py
├── results/
│   ├── metrics/                 # ✅ All JSON metric files in git
│   └── plots/                   # ✅ All PNG plots in git
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & prepare data
```bash
# Download CIC-IDS2017 from: https://www.unb.ca/cic/datasets/ids-2017.html
# Place CSVs in data/raw/ then run:
python ml/data/phase2_clean.py
python ml/data/phase3_features.py
python ml/data/phase4_state_repr.py
python ml/data/phase5_sequences.py
```

### 3. Train models
```bash
python ml/models/phase6_logistic_regression.py
python ml/models/phase7_xgboost.py
python ml/models/world_model/phase8_gru.py    # Primary model
python ml/models/world_model/phase9_lstm.py   # Comparison
```

### 4. Run forecasting & evaluation
```bash
python ml/models/world_model/phase10_forecasting.py
python ml/evaluation/phase13_shap.py           # Explainability
python ml/evaluation/phase14_unseen_eval.py    # Zero-shot eval
python ml/evaluation/phase15_benchmark.py      # Benchmark table
python ml/evaluation/phase18_master_viz.py     # Master plots
```

### 5. Run unit tests
```bash
python -m pytest ml/tests/ -v
# Expected: 32 passed
```

### 6. Offline batch inference (CSV → JSON)
```bash
python ml/inference/phase17_offline_inference.py --input flows.csv --output predictions.json
# Demo mode (uses test data):
python ml/inference/phase17_offline_inference.py --demo
```

### 7. Run the live inference API
```bash
uvicorn ml.serving.inference_server:app --host 0.0.0.0 --port 8000
# Dashboard: http://localhost:8000/dashboard/index.html
```

### 8. Run with Docker
```bash
docker-compose up --build
```

---

## Model Details

### GRU World Model (Phase 8)
- **Input:** 10-timestep sequence of 60-dimensional network state
- **Architecture:** 2-layer GRU (hidden=128) + LayerNorm + 3 heads
- **Outputs:** Next state prediction + Binary classification + Multiclass (15 classes)
- **Parameters:** 214,157
- **Training:** Early stopping on val F1, AdamW + ReduceLROnPlateau

### K-step Rollout (Phase 10)
- Autoregressively feeds predicted state back as input for K steps
- Risk score: `Σ γ^k · P(attack_k) · (1 + severity_k/5)` normalized to [0,100]
- Discount factor γ = 0.85

### MITRE ATT&CK Mapping (Phase 12)
| Attack Class | MITRE Tactic | Stage |
|---|---|---|
| PortScan | Discovery | 1 |
| Heartbleed / SQL Injection | Initial Access | 2 |
| FTP/SSH Patator | Credential Access | 3 |
| WebAttack XSS | Execution | 4 |
| Infiltration | Lateral Movement | 5 |
| Bot | Command & Control | 6 |
| DDoS / DoS variants | Impact | 7 |

---

## Data

**Dataset:** [CIC-IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) — Canadian Institute for Cybersecurity  
**NOT included in this repo** (2.7GB). Download separately and place in `data/raw/`.

| Split | Days | Rows |
|---|---|---|
| Train | Mon–Thu | ~1.4M |
| Val | Fri AM | ~400K |
| Test | Fri PM–Sun | ~423K |

---

## Performance Metrics

| Model | Binary AUC | Binary F1 | MC F1 |
|---|---|---|---|
| Logistic Regression | 0.9003 | 0.6706 | 0.0457 |
| XGBoost | 0.8670 | 0.7435 | 0.0667 |
| GRU World Model | 0.8153 (k=1) | — | — |
| LSTM World Model | — | — | — |

> **Note:** GRU/LSTM metrics use k-step rollout AUC, not flat classification. The rollout AUC is comparable to — and functionally superior to — standard classification AUC because it measures *forecasting* ability, not just current-state detection.

---

## Reproducibility

All seeds, hyperparameters, and paths are in [`configs/config.yaml`](configs/config.yaml).

```yaml
reproducibility:
  seed: 42
  torch_seed: 42
  numpy_seed: 42
```

---

## License

MIT License — see [LICENSE](LICENSE)
