"""
Phase 13 – Live Traffic Simulator
=================================
This script simulates a live network tap by reading actual test sequences
from the CIC-IDS2017 dataset and streaming them to the FastAPI prediction endpoint.

Usage:
  python traffic_simulator.py --url http://localhost:8000/predict --fps 2
"""

import time
import json
import argparse
import random
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

def main():
    parser = argparse.ArgumentParser(description="Live Traffic Simulator for OmniSentinel")
    parser.add_argument("--url", default="http://localhost:8000/predict", help="Prediction API URL")
    parser.add_argument("--fps", type=float, default=1.0, help="Flows per second to simulate")
    parser.add_argument("--seq-len", type=int, default=10, help="Sequence length")
    args = parser.parse_args()

    print(f"[SIMULATOR] Starting live traffic simulation to {args.url}")
    print(f"[SIMULATOR] Loading test dataset...")

    try:
        with open(PROC_DIR / "feature_list.json") as f:
            feat_def = json.load(f)
        feature_cols = feat_def["all_numeric_features"]
        
        # Load test set
        df_test = pd.read_parquet(PROC_DIR / "state_test.parquet")
        # Ensure chronological order
        df_test = df_test.sort_values(["day_order", "seq_idx"]).reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return

    print(f"[SIMULATOR] Dataset loaded. Starting stream at {args.fps} FPS...\n")
    print("-" * 60)

    # We will pick a random starting point in the test set to make it interesting
    # and avoid boundaries
    max_idx = len(df_test) - args.seq_len - 1000
    if max_idx < 0:
        print("Dataset too small.")
        return

    current_idx = random.randint(0, max_idx)
    sleep_time = 1.0 / args.fps

    while current_idx < len(df_test) - args.seq_len:
        # Check boundary
        window_bnd = df_test.loc[current_idx : current_idx + args.seq_len - 1, "is_boundary"].values
        if any(window_bnd):
            # Skip over boundaries
            current_idx += 1
            continue
            
        # Extract sequence features
        seq_df = df_test.loc[current_idx : current_idx + args.seq_len - 1, feature_cols]
        seq_list = seq_df.values.tolist()

        # Send to API
        payload = {
            "sequence": seq_list,
            "rollout_k": 8,
            "return_saliency": True
        }

        try:
            t0 = time.time()
            resp = requests.post(args.url, json=payload, timeout=2)
            if resp.status_code == 200:
                data = resp.json()
                t1 = time.time()
                req_ms = round((t1 - t0) * 1000, 1)

                risk = data["risk_score"]
                level = data["risk_level"]
                tactic = data["summary"]["predicted_tactic_k1"] or "BENIGN"

                color_prefix = "\033[92m" if level in ["SAFE", "LOW"] else "\033[93m" if level == "MEDIUM" else "\033[91m"
                color_suffix = "\033[0m"
                
                print(f"[{time.strftime('%H:%M:%S')}] Risk: {color_prefix}{risk:04.1f} ({level}){color_suffix} | "
                      f"Tactic: {tactic:18s} | Latency: {req_ms}ms")
            else:
                print(f"[SIMULATOR] Error {resp.status_code}: {resp.text}")
        except requests.exceptions.RequestException as e:
            print(f"[SIMULATOR] Connection error: {e}")
            print(f"[SIMULATOR] Retrying in 5 seconds...")
            time.sleep(5)
            continue

        current_idx += 1
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
