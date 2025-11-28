import pandas as pd
import numpy as np
def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df

def load_npy(npy_path):
    return np.load(npy_path, allow_pickle=True).item()
