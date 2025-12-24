import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGCN(nn.Module):
    """Heterogeneous graph convolutional network encoder."""
    
    def __init__(self, metadata, hidden_channels=64, num_layers=2, num_nodes_dict=None):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        
        # Learnable embeddings for nodes without features
        self.embeddings = nn.ModuleDict()
        
        for node_type in self.node_types:
            n_nodes = num_nodes_dict.get(node_type, 0) if num_nodes_dict else 0
            if n_nodes > 0:
                self.embeddings[node_type] = nn.Embedding(n_nodes, hidden_channels)
                nn.init.xavier_uniform_(self.embeddings[node_type].weight)
        
        # Per-node-type feature projection (LazyLinear adapts to input dim)
        # This allows different node types to have different input dims
        self.feature_projs = nn.ModuleDict({
            nt: nn.LazyLinear(hidden_channels) for nt in self.node_types
        })
        
        # LayerNorm for feature distribution stability
        self.layernorm = nn.LayerNorm(hidden_channels)
        
        # GNN layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: SAGEConv((-1, -1), hidden_channels)
                for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, x_dict, edge_index_dict):
        """
        Encode nodes to embeddings.
        
        Args:
            x_dict: Dict of node features {node_type: tensor [n_nodes, in_channels]}
            edge_index_dict: Dict of edge indices
        """
        h_dict = {}
        
        for node_type in self.node_types:
            x = x_dict.get(node_type)
            
            # Check if features exist and are non-zero
            if x is not None and x.shape[1] > 1 and x.abs().sum() > 0:
                # Use node-type specific projection
                h = self.feature_projs[node_type](x)
            else:
                # Use learnable embeddings
                h = self.embeddings[node_type].weight
            
            # Apply LayerNorm to ALL node types for consistent distribution
            h_dict[node_type] = self.layernorm(h)
        
        # GNN propagation
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                h_dict = {k: F.relu(h) for k, h in h_dict.items()}
                h_dict = {k: F.dropout(h, p=0.2, training=self.training) for k, h in h_dict.items()}
        
        return h_dict

