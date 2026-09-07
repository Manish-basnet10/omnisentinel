import pandas as pd
import numpy as np

df = pd.read_parquet("Monday-WorkingHours.pcap_ISCX.csv.parquet")
print("Data loaded")
# mimic inference server logic
df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

try:
    x = df.select_dtypes(include=[np.number]).values.astype(np.float32)
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    # find which column has the issue
    for col in df.select_dtypes(include=[np.number]).columns:
        try:
            df[col].values.astype(np.float32)
        except Exception as e2:
            print(f"Col {col} error: {e2}")
            print(df[col].max(), df[col].min())

