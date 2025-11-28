import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import SAGEConv, BatchNorm

class ImprovedGCNRegressor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        
        # GraphSAGE layers (no edge_dim support)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output head
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.bn_out = BatchNorm(hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.norm = LayerNorm(hidden_channels)
    
    def forward(self, x, edge_index, edge_attr=None):
        # Initial residual connection
        residual = x
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)  # No edge_attr for SAGEConv
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Residual connection for first layer only
            if i == 0 and residual.shape == x.shape:
                x = x + residual[:x.shape[0]]
        
        x = self.norm(x)
        x = F.relu(self.bn_out(self.lin1(x)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        out = self.lin2(x).squeeze(-1)
        return out
