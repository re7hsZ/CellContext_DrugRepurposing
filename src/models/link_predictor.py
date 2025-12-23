import torch
import torch.nn as nn

class LinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_src, x_dst, edge_label_index):
        """
        Computes score for edges in edge_label_index.
        x_src: Embeddings of source nodes (e.g. drugs) [Num_Drugs, Hidden]
        x_dst: Embeddings of destination nodes (e.g. diseases) [Num_Diseases, Hidden]
        edge_label_index: [2, Num_Edges]
        """
        # Select embeddings
        row, col = edge_label_index
        
        # Check bounds (debug helper)
        if row.max() >= x_src.size(0) or col.max() >= x_dst.size(0):
             # This might happen if validation edges contain nodes not in training set (if strict split),
             # but random link split usually keeps nodes.
             pass

        src_emb = x_src[row]
        dst_emb = x_dst[col]
        
        # Dot product
        # Shape: [Num_Edges]
        score = (src_emb * dst_emb).sum(dim=-1)
        
        return score
