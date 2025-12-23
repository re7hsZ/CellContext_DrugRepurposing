import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv


class HeteroGCN(nn.Module):
    """Heterogeneous graph convolutional network encoder."""
    
    def __init__(self, metadata, hidden_channels=64, num_layers=2, 
                 num_nodes_dict=None, input_channels_dict=None):
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        
        # Input projection layers for nodes with features (e.g., gene with PINNACLE)
        self.input_proj = nn.ModuleDict()
        
        # Learnable embeddings for nodes without features (e.g., drug, disease)
        self.embeddings = nn.ModuleDict()
        
        for node_type in self.node_types:
            n_nodes = num_nodes_dict.get(node_type, 0) if num_nodes_dict else 0
            in_channels = input_channels_dict.get(node_type, 0) if input_channels_dict else 0
            
            if in_channels > 0:
                # Use linear projection for nodes with input features
                self.input_proj[node_type] = nn.Linear(in_channels, hidden_channels)
            
            if n_nodes > 0:
                # Learnable embeddings as fallback or for nodes without features
                self.embeddings[node_type] = nn.Embedding(n_nodes, hidden_channels)
                nn.init.xavier_uniform_(self.embeddings[node_type].weight)
        
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
        # Build initial representations
        h_dict = {}
        for node_type in self.node_types:
            x = x_dict.get(node_type)
            
            # Check if input features are available and non-zero
            if x is not None and node_type in self.input_proj:
                # Check if features are meaningful (not all zeros)
                if x.abs().sum() > 0:
                    h_dict[node_type] = self.input_proj[node_type](x)
                else:
                    # Features are all zeros, use learnable embeddings
                    h_dict[node_type] = self.embeddings[node_type].weight
            elif node_type in self.embeddings:
                # No input features, use learnable embeddings
                h_dict[node_type] = self.embeddings[node_type].weight
        
        # Apply GNN layers
        for i, conv in enumerate(self.convs):
            h_dict = conv(h_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                h_dict = {k: F.relu(h) for k, h in h_dict.items()}
                h_dict = {k: F.dropout(h, p=0.2, training=self.training) for k, h in h_dict.items()}
        
        return h_dict
