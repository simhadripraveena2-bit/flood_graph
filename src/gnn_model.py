import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class FloodGNN(torch.nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def build_pyg_data(df_rain, df_inflow, G):
    """
    Builds PyTorch Geometric Data object:
    Nodes: unique (lat, lon)
    Node features: mean rainfall intensity per node
    Target: inflow (mean per day or total)
    """
    coords = df_rain[["latitude", "longitude"]].drop_duplicates().reset_index(drop=True)
    node_map = {tuple(v): i for i, v in coords.iterrows()}

    # Feature: mean rainfall intensity per node
    df_node = (
        df_rain.groupby(["latitude", "longitude"])["intensity"]
        .mean()
        .reset_index()
    )
    x = torch.tensor(df_node["intensity"].values, dtype=torch.float32).view(-1, 1)

    # Edge index from NetworkX
    edges = list(G.edges())
    edge_idx = torch.tensor([[node_map[u], node_map[v]] for u, v in edges], dtype=torch.long).t().contiguous()

    # Target variable = inflow mean
    y = torch.tensor(df_inflow["inflow"].values, dtype=torch.float32)
    y = y[:len(x)] if len(y) > len(x) else torch.nn.functional.pad(y, (0, len(x) - len(y)))

    data = Data(x=x, edge_index=edge_idx, y=y)
    return data


def evaluate_regression(y_true, y_pred):
    """
    Returns regression metrics: MAE, RMSE, R2
    """
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def train_gnn(df_rain, df_inflow, G, epochs=100, lr=1e-3):
    """
    Train GNN on rainfall graph to predict inflow values.
    Returns: model, metrics, and data
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = build_pyg_data(df_rain, df_inflow, G).to(device)
    model = FloodGNN(input_dim=1, hidden_dim=32, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(data.x, data.edge_index).squeeze()
    metrics = evaluate_regression(data.y, preds)

    print("GNN Training Completed")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"R²:   {metrics['R2']:.4f}")

    return model, metrics, data
