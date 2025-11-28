import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import ImprovedGCNRegressor
from utils import temporal_split, YScaler, compute_metrics
from graph_builder import build_spatio_temporal_graph
import os

def filter_and_remap_edges(edge_index_np, keep_idx):
    """Filter and remap edges for subset of nodes"""
    if edge_index_np.size == 0:
        return np.zeros((2, 0), dtype=int)
    
    keep_set = set(keep_idx.tolist())
    # Keep edges where BOTH nodes are in keep_set
    mask = np.array([u in keep_set and v in keep_set 
                    for u, v in edge_index_np.T])
    
    if not np.any(mask):
        return np.zeros((2, 0), dtype=int)
    
    filtered_edges = edge_index_np[:, mask]
    
    # Remap node indices to 0...len(keep_idx)-1
    old_to_new = {old: new for new, old in enumerate(keep_idx)}
    remapped = np.array([[old_to_new.get(u, u), old_to_new.get(v, v)] 
                        for u, v in filtered_edges.T]).T
    
    return remapped.astype(np.int64)

def build_pyg_data(arr, idx):
    """Build PyG Data object for node subset"""
    keep_idx = np.array(idx, dtype=int)
    
    x = arr['x'][keep_idx].astype(np.float32)
    y = arr['y_scaled'][keep_idx].astype(np.float32) if 'y_scaled' in arr else arr['y'][keep_idx].astype(np.float32)
    
    edge_index = filter_and_remap_edges(arr['edge_index'], keep_idx)
    
    return {
        'x': torch.tensor(x, dtype=torch.float),
        'edge_index': torch.tensor(edge_index, dtype=torch.long),
        'y': torch.tensor(y, dtype=torch.float)
    }

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

def train_pipeline(csv_path="processed_long_rainfall_v2.csv", epochs=500, lr=1e-3, 
                   device=None, model_dir="models"):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Using device: {device}")
    print(f"Loading data from: {csv_path}")
    
    # Build enhanced graph
    arr, df = build_spatio_temporal_graph(csv_path)
    print(f"Dataset: {arr['x'].shape[0]} nodes, {arr['x'].shape[1]} features")
    print(f"Edges: {arr['edge_index'].shape[1] if arr['edge_index'].size > 0 else 0}")
    
    # Temporal split (time-respecting)
    train_idx, val_idx, test_idx = temporal_split(df)
    print(f"Splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Scale target variable
    scaler = YScaler()
    scaler.fit(arr['y'])
    arr['y_scaled'] = scaler.transform(arr['y'])
    
    # Save everything
    os.makedirs(model_dir, exist_ok=True)
    np.save(os.path.join(model_dir, "processed_data.npy"), arr)
    np.save(os.path.join(model_dir, "split_indices.npy"), {
        'train': train_idx, 'val': val_idx, 'test': test_idx
    })
    np.save(os.path.join(model_dir, "scaler.npy"), {
        'mean': scaler.mean_, 'std': scaler.std_
    })
    
    device = torch.device(device)
    
    # Create data objects
    train_data = build_pyg_data(arr, train_idx)
    val_data = build_pyg_data(arr, val_idx)
    
    train_data = {k: v.to(device) for k, v in train_data.items()}
    val_data = {k: v.to(device) for k, v in val_data.items()}
    
    # Initialize model (fixed - no edge_dim)
    model = ImprovedGCNRegressor(in_channels=arr['x'].shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)  # FIXED: removed verbose
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_path = os.path.join(model_dir, "best_gcn_reg_improved.pt")
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(train_data['x'], train_data['edge_index'])
        loss = F.mse_loss(out, train_data['y'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Validation every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data['x'], val_data['edge_index'])
                val_loss = F.mse_loss(val_out, val_data['y']).item()
            
            # FIXED: Manual LR change detection
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                print(f"  -> LR reduced: {old_lr:.2e} → {new_lr:.2e}")
            
            print(f"Epoch {epoch:3d} | Train MSE: {loss.item():.4f} | Val MSE: {val_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'model_state': model.state_dict(),
                    'scaler': {'mean': scaler.mean_, 'std': scaler.std_},
                    'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
                    'optimizer_state': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, best_path)
                print(f"  -> New best model saved! Val MSE: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 50:
                    print(f"Early stopping at epoch {epoch}")
                    break
    
    print(f"Training complete! Best Val MSE: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_path}")
    return best_path

if __name__ == "__main__":
    train_pipeline("notebooks/processed_long_rainfall_v2.csv")
