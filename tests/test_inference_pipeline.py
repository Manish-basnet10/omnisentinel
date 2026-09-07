import os
import time
import requests
import pandas as pd
from pathlib import Path
from ml.serving.feature_contract import PARTIAL_MODEL_FEATURES, ORIGINAL_FEATURES

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

def generate_test_files():
    print("Generating test files from cleaned_full.parquet...")
    df = pd.read_parquet(PROC_DIR / "cleaned_full.parquet").head(100)
    
    # Needs to be engineered to have 60 features. Wait, cleaned_full only has 46 + meta!
    # I should read features_engineered.parquet to get 60!
    try:
        df_eng = pd.read_parquet(PROC_DIR / "features_engineered.parquet").head(100)
        df_eng.to_csv("test_60.csv", index=False)
        print("Created test_60.csv (has 60+ features)")
    except Exception as e:
        print("Could not create test_60.csv", e)
        df_eng = df.copy() # fallback
        
    df[ORIGINAL_FEATURES].to_csv("test_46.csv", index=False)
    print("Created test_46.csv (has 46 features)")
    
    df[PARTIAL_MODEL_FEATURES].to_csv("test_partial.csv", index=False)
    print("Created test_partial.csv (has 20 features)")


def test_endpoint(filename, expected_mode, expected_available):
    url = "http://127.0.0.1:8000/api/analyze-pcap"
    
    session = requests.Session()
    register_data = {"email": "test@test.com", "username": "testuser_inference", "password": "password"}
    session.post("http://127.0.0.1:8000/auth/register", json=register_data)
    
    # login expects application/x-www-form-urlencoded
    login_data = {"username": "test@test.com", "password": "password"}
    login_resp = session.post("http://127.0.0.1:8000/auth/login", data=login_data)
    
    if login_resp.status_code == 200:
        token = login_resp.json().get("access_token")
        headers = {"Authorization": f"Bearer {token}"}
    else:
        print("Login failed, trying without auth.")
        headers = {}
    
    print(f"\n--- Testing {filename} ---")
    with open(filename, "rb") as f:
        files = {"file": (filename, f, "text/csv")}
        resp = requests.post(url, headers=headers, files=files)
        
    if resp.status_code == 200:
        data = resp.json()
        print(f"Success! Model Mode: {data.get('model_mode')}")
        print(f"Feature Coverage: {data.get('feature_coverage')}")
        
        assert data["model_mode"] == expected_mode, f"Expected {expected_mode} but got {data['model_mode']}"
        assert data["feature_coverage"]["available"] == expected_available, f"Expected {expected_available} available features, got {data['feature_coverage']['available']}"
        print("Assertions passed.")
    else:
        print(f"Error {resp.status_code}: {resp.text}")
        assert False, "Endpoint returned error"


if __name__ == "__main__":
    generate_test_files()
    time.sleep(2) # ensure backend is up
    test_endpoint("test_60.csv", "gru_60", 60)
    test_endpoint("test_46.csv", "gru_60", 46)
    test_endpoint("test_partial.csv", "partial_pytorch", len(PARTIAL_MODEL_FEATURES))
    print("\nAll tests passed successfully!")
