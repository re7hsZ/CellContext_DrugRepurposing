import torch.nn as nn


class LinkPredictor(nn.Module):
    """Dot-product link predictor for drug-disease edges."""
    
    def forward(self, x_src, x_dst, edge_index):
        """Compute edge scores via dot product."""
        src_emb = x_src[edge_index[0]]
        dst_emb = x_dst[edge_index[1]]
        return (src_emb * dst_emb).sum(dim=-1)
