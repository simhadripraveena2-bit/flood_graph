# scripts/run_rf_improved_small_dataset.py
"""
Improved Random Forest for Small Dataset Flood Prediction
Optimized for 54 samples: conservative hyperparameters, no SMOTE, better CV
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score)
import joblib
import warnings
warnings.filterwarnings('ignore')

OUT = Path("flood_results")
OUT.mkdir(exist_ok=True)

print("Improved Random Forest for Small Dataset (54 samples)")
print("="*60)

# Load and preprocess (same as original)
df = pd.read_csv('notebooks/processed_long_rainfall_v2.csv', parse_dates=['date'])
daily = df.groupby("date").agg({
    "rainfall": "sum", "inflow": "first", "lat": "mean", "lon": "mean"
}).reset_index().rename(columns={"rainfall": "rain_total"})

daily['inflow_next5'] = daily['inflow'].shift(-5)
daily = daily.dropna(subset=['inflow_next5']).reset_index(drop=True)
daily['flood_next5'] = (daily['inflow_next5'] > daily['inflow'].quantile(0.75)).astype(int)

print(f"Dataset size: {len(daily)} samples")

# Same feature engineering
features = []
for lag in [1,2,3,5,7]:
    daily[f"rain_lag_{lag}"] = daily['rain_total'].shift(lag).fillna(0)
    features.append(f"rain_lag_{lag}")

for w in [3,7]:
    daily[f"rain_roll_sum_{w}"] = daily['rain_total'].rolling(w).sum().fillna(0)
    daily[f"rain_roll_max_{w}"] = daily['rain_total'].rolling(w).max().fillna(0)
    features.extend([f"rain_roll_sum_{w}", f"rain_roll_max_{w}"])

daily['rain_acceleration'] = daily['rain_total'] - daily['rain_total'].shift(1).fillna(0)
daily['rain_intensity'] = daily['rain_total'] / (daily['rain_roll_sum_7'] + 1)
features.extend(['rain_acceleration', 'rain_intensity'])

for lag in [1,3]:
    daily[f"inflow_lag_{lag}"] = daily['inflow'].shift(lag).fillna(daily['inflow'].mean())
    features.append(f"inflow_lag_{lag}")

daily['dayofyear_sin'] = np.sin(2 * np.pi * daily['date'].dt.dayofyear / 365.25)
daily['month_sin'] = np.sin(2 * np.pi * daily['date'].dt.month / 12)
features.extend(['dayofyear_sin', 'month_sin'])

daily['heavy_rain'] = (daily['rain_total'] > daily['rain_total'].quantile(0.9)).astype(int)
features.append('heavy_rain')

print(f"Total features (reduced): {len(features)}")

# Time-based split (80/20)
split_idx = int(0.8 * len(daily))
train_df = daily.iloc[:split_idx].reset_index(drop=True)
test_df = daily.iloc[split_idx:].reset_index(drop=True)

X_train = train_df[features].fillna(0)
y_train_reg = train_df['inflow_next5'].values
y_train_clf = train_df['flood_next5'].values

X_test = test_df[features].fillna(0)
y_test_reg = test_df['inflow_next5'].values
y_test_clf = test_df['flood_next5'].values

print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

# CONSERVATIVE HYPERPARAMETERS FOR SMALL DATASET [web:2][web:4][web:8]
tscv = TimeSeriesSplit(n_splits=3)

# Regression: Conservative params for small data
reg_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],           # Shallow trees
    'min_samples_split': [5, 10],     # Higher to prevent overfitting
    'min_samples_leaf': [2, 4],       # Higher leaf minimum
    'max_features': ['sqrt', 'log2']  # Feature subsampling
}

print("\nTuning Regression model...")
regressor = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_reg = GridSearchCV(regressor, reg_param_grid, cv=tscv, 
                       scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
grid_reg.fit(X_train, y_train_reg)
print(f"Best regression params: {grid_reg.best_params_}")
print(f"Best CV RMSE: {-grid_reg.best_score_**0.5:.2f}")
best_rf_reg = grid_reg.best_estimator_

# Classification: NO SMOTE, use class_weight instead [web:2]
clf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced']      # Handle imbalance internally
}

print("\nTuning Classification model (NO SMOTE)...")
classifier = RandomForestClassifier(random_state=42)
grid_clf = GridSearchCV(classifier, clf_param_grid, cv=tscv, 
                       scoring='roc_auc', n_jobs=-1, verbose=1)
grid_clf.fit(X_train, y_train_clf)
print(f"Best classification params: {grid_clf.best_params_}")
print(f"Best CV AUC: {grid_clf.best_score_:.3f}")
best_rf_clf = grid_clf.best_estimator_

# Test predictions
y_pred_reg = best_rf_reg.predict(X_test)
y_pred_clf_proba = best_rf_clf.predict_proba(X_test)[:, 1]
y_pred_clf = best_rf_clf.predict(X_test)

# Comprehensive metrics
metrics = {
    "regression": {
        "test_r2": float(r2_score(y_test_reg, y_pred_reg)),
        "test_mae": float(mean_absolute_error(y_test_reg, y_pred_reg)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
    },
    "classification": {
        "test_accuracy": float(accuracy_score(y_test_clf, y_pred_clf)),
        "test_precision": float(precision_score(y_test_clf, y_pred_clf, zero_division=0)),
        "test_recall": float(recall_score(y_test_clf, y_pred_clf, zero_division=0))
    },
    "best_params": {
        "regression": grid_reg.best_params_,
        "classification": grid_clf.best_params_
    }
}

print("\n" + "="*50)
print("IMPROVED FINAL EVALUATION METRICS:")
print(json.dumps(metrics, indent=2))

# Save everything
joblib.dump(best_rf_reg, OUT / "rf_regressor.pkl")
joblib.dump(best_rf_clf, OUT / "rf_classifier.pkl")
with open(OUT / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
daily.to_csv(OUT / "features.csv", index=False)

# Enhanced visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Regression plot
axes[0,0].plot(test_df["date"], y_test_reg, 'b-o', label="Actual", markersize=4)
axes[0,0].plot(test_df["date"], y_pred_reg, 'r-s', label="Predicted", markersize=4)
axes[0,0].set_title(f"Regression: R²={metrics['regression']['test_r2']:.3f}\nMAE={metrics['regression']['test_mae']:.0f}")
axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=45)

# Classification plot
axes[0,1].plot(test_df["date"], y_test_clf, 'b-o', label="Actual Flood", markersize=4)
axes[0,1].plot(test_df["date"], y_pred_clf, 'r-s', label="Predicted Flood", markersize=4)
axes[0,1].set_title(f"Classification: Accuracy={metrics['classification']['test_accuracy']:.3f}")
axes[0,1].legend()
axes[0,1].tick_params(axis='x', rotation=45)

# Feature importance (Regression)
feat_imps_reg = pd.Series(best_rf_reg.feature_importances_, index=features).sort_values(ascending=False)[:10]
feat_imps_reg.plot(kind='barh', ax=axes[1,0])
axes[1,0].set_title("Top 10 Regression Feature Importances")
axes[1,0].invert_yaxis()

# Feature importance (Classification)
feat_imps_clf = pd.Series(best_rf_clf.feature_importances_, index=features).sort_values(ascending=False)[:10]
feat_imps_clf.plot(kind='barh', ax=axes[1,1])
axes[1,1].set_title("Top 10 Classification Feature Importances")
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig(OUT / "rf_results.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\nAll saved to {OUT.resolve()}")
