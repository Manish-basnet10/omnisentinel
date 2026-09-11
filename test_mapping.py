import pandas as pd
import json
import asyncio
from pathlib import Path
from ml.serving.inference_server import _analyze_file_sync, load_model
from ml.serving.feature_contract import ORIGINAL_FEATURES

# Read actual CIC-IDS2018 dataset (first 50 rows)
csv_path = "/Users/manishbasnet/Downloads/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv"

def test():
    load_model()
    print("Testing real CIC-IDS2018 CSV mapping...")
    # Mocking pd.read_csv to read fewer rows would require mocking inside inference_server, 
    # but the server reads the whole file. We'll just run it against the full file if it's small enough,
    # but 204MB takes a moment. Let's just create a small chunk of it and test against that.
    df = pd.read_csv(csv_path, nrows=50)
    df.to_csv("test_cicflowmeter_chunk.csv", index=False)
    
    res = _analyze_file_sync(Path("test_cicflowmeter_chunk.csv"), "test_cicflowmeter_chunk.csv")
    print(f"Status: {res['status']}")
    print(f"Model mode chosen: {res['model_mode']}")
    print(f"Engineered features count: {res['feature_coverage']['engineered']}")
    print(f"Final features count: {res['feature_coverage']['final']}")
    print("Forecast output:")
    print(json.dumps(res.get('forecast', []), indent=2))
    
if __name__ == "__main__":
    test()
