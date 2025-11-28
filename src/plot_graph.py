import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Load GNN metrics ----
with open("models/dashboard_metrics.json", "r") as f:
    gnn_json = json.load(f)

gnn_metrics = {
    "MAE": float(gnn_json['mae'].replace(" mm","")),
    "RMSE": float(gnn_json['rmse'].replace(" mm","")),
    "R2": float(gnn_json['r2'])
}

# ---- GNN Metrics Bar Chart ----
metrics = ["MAE", "RMSE", "R2"]
values = [gnn_metrics[m] for m in metrics]

plt.figure(figsize=(6,4))
bars = plt.bar(metrics, values, color='lightgreen')
plt.title("GNN Model Metrics")
plt.ylabel("Value")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.4f}", ha='center')
plt.tight_layout()
plt.savefig("models/gnn_metrics.png")
plt.show()

# ---- Predicted vs Actual Scatter Plot ----
try:
    preds_dict = np.load("models/test_predictions.npy", allow_pickle=True).item()
    y_true = preds_dict['y_true']
    y_pred = preds_dict['y_pred']

    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='teal')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Inflow")
    plt.ylabel("Predicted Inflow")
    plt.title("GNN Predicted vs Actual Inflow")
    plt.tight_layout()
    plt.savefig("models/gnn_pred_vs_actual.png")
    plt.show()
except FileNotFoundError:
    print("test_predictions.npy not found; skipping scatter plot.")
