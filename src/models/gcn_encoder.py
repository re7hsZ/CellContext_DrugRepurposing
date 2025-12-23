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
        
        self.embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            n_nodes = num_nodes_dict.get(node_type, 0) if num_nodes_dict else 0
            if n_nodes > 0:
                self.embeddings[node_type] = nn.Embedding(n_nodes, hidden_channels)
                nn.init.xavier_uniform_(self.embeddings[node_type].weight)
        
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                et: SAGEConv((-1, -1), hidden_channels)
                for et in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, edge_index_dict):
        """Encode nodes to embeddings."""
        x_dict = {nt: emb.weight for nt, emb in self.embeddings.items()}
        
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            if i < len(self.convs) - 1:
                x_dict = {k: F.relu(x) for k, x in x_dict.items()}
                x_dict = {k: F.dropout(x, p=0.2, training=self.training) for k, x in x_dict.items()}
        
        return x_dict
