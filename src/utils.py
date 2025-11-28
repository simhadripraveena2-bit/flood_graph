import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}

def temporal_split(df, train_ratio=0.7, val_ratio=0.2, seed=42):
    """Temporal split respecting time order"""
    n = len(df)
    np.random.seed(seed)
    split1 = int(n * train_ratio)
    split2 = int(n * (train_ratio + val_ratio))
    
    train_idx = np.arange(split1)
    val_idx = np.arange(split1, split2)
    test_idx = np.arange(split2, n)
    
    return train_idx, val_idx, test_idx

class YScaler:
    def __init__(self):
        self.mean_ = 0.0
        self.std_ = 1.0
    
    def fit(self, y):
        self.mean_ = float(np.mean(y))
        self.std_ = float(np.std(y)) if np.std(y) > 0 else 1.0
    
    def transform(self, y):
        return (y - self.mean_) / self.std_
    
    def inverse_transform(self, y):
        return y * self.std_ + self.mean_
