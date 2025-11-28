import json
import numpy as np
import torch
import torch.nn.functional as F
from model import ImprovedGCNRegressor
from utils import compute_metrics
import os

def filter_and_remap_edges(edge_index_np, keep_idx):
    """Filter and remap edges for subset of nodes"""
    if edge_index_np.size == 0:
        return np.zeros((2, 0), dtype=int)
    
    keep_set = set(keep_idx.tolist())
    mask = np.array([u in keep_set and v in keep_set 
                    for u, v in edge_index_np.T])
    
    if not np.any(mask):
        return np.zeros((2, 0), dtype=int)
    
    filtered_edges = edge_index_np[:, mask]
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    remapped = np.array([[old_to_new.get(u, u), old_to_new.get(v, v)] 
                        for u, v in filtered_edges.T]).T
    
    return remapped.astype(np.int64)

def build_pyg_data(arr, idx):
    """Build PyG Data object for node subset"""
    keep_idx = np.array(idx, dtype=int)
    
    x = arr['x'][keep_idx].astype(np.float32)
    y = arr['y'][keep_idx].astype(np.float32)  # Use original y for evaluation
    
    edge_index = filter_and_remap_edges(arr['edge_index'], keep_idx)
    
    return {
        'x': torch.tensor(x, dtype=torch.float),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'y': torch.tensor(y, dtype=torch.float)
    }

def safe_load_checkpoint(ckpt_path, device):
    """Safely load checkpoint with PyTorch 2.6+ weights_only fix"""
    try:
        # PyTorch 2.6+ default is weights_only=True, use False for full checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        print("Checkpoint loaded with weights_only=False (full checkpoint)")
        return ckpt
    except Exception as e:
        print(f"Error loading with weights_only=False: {e}")
        print("Trying alternative loading method...")
        # Fallback: load as dict and extract model_state
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            return ckpt
        raise RuntimeError("Cannot load checkpoint safely")

def evaluate(csv_path="processed_long_rainfall_v2.csv", 
             ckpt_path="models/best_gcn_reg_improved.pt", 
             model_dir="models"):
    print(f"Loading processed data from: {model_dir}/processed_data.npy")
    
    # Load processed data
    arr = np.load(os.path.join(model_dir, "processed_data.npy"), allow_pickle=True).item()
    
    # Safely load splits with key checking
    try:
        splits = np.load(os.path.join(model_dir, "split_indices.npy"), allow_pickle=True).item()
        print("Available split keys:", list(splits.keys()))
        
        # Handle different possible key names
        if 'test_idx' in splits:
            test_idx = splits['test_idx']
        elif 'test' in splits:
            test_idx = splits['test']
        else:
            # Fallback: use last 10% as test
            n = len(arr['y'])
            test_idx = np.arange(int(0.9 * n), n)
            print(f"No test split found, using last {len(test_idx)} samples as test")
    except FileNotFoundError:
        print("No split_indices.npy found, using last 10% as test set")
        n = len(arr['y'])
        test_idx = np.arange(int(0.9 * n), n)
    
    # Load scaler (fallback if missing)
    try:
        scaler_params = np.load(os.path.join(model_dir, "scaler.npy"), allow_pickle=True).item()
        mean_y, std_y = scaler_params['mean'], scaler_params['std']
    except FileNotFoundError:
        print("No scaler.npy found, assuming unscaled data (no inverse transform)")
        mean_y, std_y = 0.0, 1.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Test set: {len(test_idx)} nodes")
    
    # Load test data (use scaled y if available, else original)
    if 'y_scaled' in arr:
        arr['y_test_scaled'] = arr['y_scaled'][test_idx]
    test_data = build_pyg_data(arr, test_idx)
    test_data = {k: v.to(device) for k, v in test_data.items()}
    
    # Initialize model
    model = ImprovedGCNRegressor(in_channels=arr['x'].shape[1]).to(device)
    
    # FIXED: Safe checkpoint loading for PyTorch 2.6+
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = safe_load_checkpoint(ckpt_path, device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    
    print("Making predictions...")
    with torch.no_grad():
        preds_scaled = model(test_data['x'], test_data['edge_index']).cpu().numpy()
    
    # Inverse transform predictions if scaled
    if std_y != 1.0:
        preds = preds_scaled * std_y + mean_y
    else:
        preds = preds_scaled
    
    y_true = arr['y'][test_idx]
    
    # Compute metrics
    metrics = compute_metrics(y_true, preds)
    
    print("\n" + "="*60)
    print("IMPROVED GCN REGRESSOR RESULTS")
    print("="*60)
    print(f"Test Set Size:  {len(y_true):4d} samples")
    print(f"MAE:            {metrics['mae']:8.4f}")
    print(f"RMSE:           {metrics['rmse']:8.4f}")
    print(f"R² Score:       {metrics['r2']:8.4f}")
    print("="*60)
    
    # Compare with baseline (mean predictor)
    baseline_mse = np.mean((y_true - np.mean(y_true))**2)
    baseline_rmse = np.sqrt(baseline_mse)
    print(f"Baseline RMSE:  {baseline_rmse:8.4f}")
    print(f"Improvement:    {((baseline_rmse - metrics['rmse'])/baseline_rmse*100):+.1f}%")
    
    # Save results
    results = {
        'y_true': y_true,
        'y_pred': preds,
        'metrics': metrics,
        'test_idx': test_idx,
        'mean_y': mean_y,
        'std_y': std_y
    }
    np.save(os.path.join(model_dir, "test_predictions.npy"), results)
    print(f"\nPredictions saved to: {model_dir}/test_predictions.npy")

    dashboard_metrics = {
        "r2": f"{metrics['r2']:.4f}",
        "mae": f"{metrics['mae']:.4f} mm",
        "rmse": f"{metrics['rmse']:.4f} mm",
        "improvement": f"{((baseline_rmse - metrics['rmse'])/baseline_rmse*100):+.1f}%",
        "r2_delta": f"↑ +{metrics['r2']*100:.1f}%",
        "test_size": len(y_true),
        "baseline_rmse": f"{baseline_rmse:.4f} mm"
    }
    with open(os.path.join(model_dir, "dashboard_metrics.json"), "w") as f:
        json.dump(dashboard_metrics, f, indent=2)
    print(f"Dashboard metrics saved to {model_dir}/dashboard_metrics.json")
    
    return metrics

if __name__ == "__main__":
    evaluate()
