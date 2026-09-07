#!/bin/bash
# OmniSentinel Startup Script
# Run this ONCE to pre-warm PyTorch and start all 3 services.
# Usage: bash start.sh

set -e
PROJ="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJ/venv/bin/python"
cd "$PROJ"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   OmniSentinel – Starting Up...      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── STEP 1: Pre-warm PyTorch (compiles ARM kernels once, cached after) ────────
echo "[1/3] Pre-warming PyTorch (this takes ~60s the very first time)..."
"$PYTHON" -c "
import torch, warnings
warnings.filterwarnings('ignore')
x = torch.randn(1, 10, 60)
print('    ✓ PyTorch', torch.__version__, 'ready on', ('Apple Silicon' if torch.backends.mps.is_available() else 'CPU'))
"
echo ""

# ── STEP 2: Start FastAPI backend ─────────────────────────────────────────────
echo "[2/3] Starting FastAPI backend on http://localhost:8000 ..."
"$PYTHON" -m uvicorn ml.serving.inference_server:app \
    --host 0.0.0.0 --port 8000 --log-level info &
BACKEND_PID=$!
echo "    Backend PID: $BACKEND_PID"

# Wait until /health responds (max 120 seconds)
echo "    Waiting for backend to be ready..."
for i in $(seq 1 60); do
    sleep 2
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "    ✓ Backend is UP!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "    ✗ Backend did not start in 120 seconds. Check for errors above."
        exit 1
    fi
    echo "    ... still loading ($((i*2))s)"
done
echo ""

# ── STEP 3: Start Traffic Simulator ───────────────────────────────────────────
echo "[3/3] Starting Traffic Simulator..."
"$PYTHON" ml/serving/traffic_simulator.py &
SIM_PID=$!
echo "    Simulator PID: $SIM_PID"
echo ""

# ── STEP 4: Start Frontend ────────────────────────────────────────────────────
echo "[4/4] Starting Frontend (Vite)..."
cd "$PROJ/frontend"
npm run dev &
FRONTEND_PID=$!
cd "$PROJ"
echo "    Frontend PID: $FRONTEND_PID"
echo ""

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  ✓ All services started!                                 ║"
echo "║                                                          ║"
echo "║  Dashboard:  http://localhost:5173                       ║"
echo "║  API:        http://localhost:8000                       ║"
echo "║  API Docs:   http://localhost:8000/api/docs              ║"
echo "║                                                          ║"
echo "║  Press Ctrl+C to stop all services                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Wait and forward Ctrl+C to all children
trap "echo ''; echo 'Stopping all services...'; kill $BACKEND_PID $SIM_PID $FRONTEND_PID 2>/dev/null; exit 0" INT TERM
wait
