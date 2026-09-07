import requests
import time
from pathlib import Path
import pandas as pd

BASE_URL = "http://localhost:8000"

def get_token():
    try:
        # Register if not exists
        requests.post(f"{BASE_URL}/auth/register", json={
            "email": "test@test.com", "password": "password", "name": "Test"
        })
        # Login using form data
        r = requests.post(f"{BASE_URL}/auth/login", data={
            "username": "test@test.com", "password": "password"
        })
        return r.json().get("access_token")
    except Exception as e:
        print("Login failed", e)
        return None

token = get_token()
headers = {"Authorization": f"Bearer {token}"} if token else {}

src_parquet = Path("data/processed/state_test.parquet")
target_parquet = Path("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet")
target_csv = Path("test_data.csv")

if src_parquet.exists():
    df = pd.read_parquet(src_parquet).head(100)
    df.to_parquet(target_parquet)
    df.to_csv(target_csv, index=False)
else:
    print("Source parquet not found")
    exit(1)

for fpath in [target_parquet, target_csv]:
    print(f"\nUploading {fpath}...")
    with open(fpath, "rb") as f:
        files = {"file": (fpath.name, f, "application/octet-stream")}
        res = requests.post(f"{BASE_URL}/api/analyze-pcap", files=files, headers=headers)
        if res.status_code == 200:
            print("Success! Risk score:", res.json()["current_state"]["risk_score"])
        else:
            print("Failed!", res.status_code, res.text)
            exit(1)

print("\nAll formats tested successfully.")
