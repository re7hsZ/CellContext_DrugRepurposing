import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

class HeteroGCN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=64, num_layers=2, num_nodes_dict=None):
        """
        metadata: Tuple (node_types, edge_types) from data.metadata()
        num_nodes_dict: Dictionary {node_type: num_nodes} for embedding initialization.
        """
        super().__init__()
        self.node_types, self.edge_types = metadata
        self.hidden_channels = hidden_channels
        
        # 1. Learnable Embeddings
        # We need embeddings for all node types.
        if num_nodes_dict is None:
            raise ValueError("num_nodes_dict must be provided to initialize embeddings.")
            
        self.embeddings = nn.ModuleDict()
        for node_type in self.node_types:
            num_nodes = num_nodes_dict.get(node_type, 0)
            if num_nodes > 0:
                self.embeddings[node_type] = nn.Embedding(num_nodes, hidden_channels)
                nn.init.xavier_uniform_(self.embeddings[node_type].weight)
        
        # 2. GNN Layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # We use SAGEConv which works well for bipartite/hetero graphs
            # (-1, -1) means lazy initialization of input shapes, handling different feature sizes if needed
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in self.edge_types
            }, aggr='sum')
            self.convs.append(conv)

    def forward(self, edge_index_dict):
        """
        edge_index_dict: Dictionary of edge indices.
        Returns: dictionary of node embeddings {node_type: tensor}.
        """
        # 1. Get initial embeddings
        x_dict = {
            node_type: emb.weight 
            for node_type, emb in self.embeddings.items()
        }
        
        # 2. Propagate
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            
            # Activation (except last layer? Usually yes for GCN embeddings)
            # But here we output embeddings for LinkPrediction, so maybe last layer needs activation too, 
            # or we keep it linear for dot product. 
            # Standard GCN uses ReLU between layers.
            if i < len(self.convs) - 1:
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}
                x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
        
        return x_dict
