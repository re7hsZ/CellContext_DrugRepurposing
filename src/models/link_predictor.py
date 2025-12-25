import torch
import torch.nn as nn
import torch.nn.functional as F


class LinkPredictor(nn.Module):
    """Link predictor with optional similarity-based zero-shot mode (TxGNN-style)."""
    
    def __init__(self, hidden_channels=None, use_sim_decoder=False):
        super().__init__()
        self.use_sim_decoder = use_sim_decoder
        
        if use_sim_decoder and hidden_channels:
            # Learnable similarity projection for SimGNN-style decoder
            self.sim_proj = nn.Linear(hidden_channels, hidden_channels)
            self.temp = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x_src, x_dst, edge_index, 
                train_disease_emb=None, train_drug_scores=None):
        """
        Compute edge scores.
        
        Args:
            x_src: Drug embeddings [n_drugs, hidden]
            x_dst: Disease embeddings [n_diseases, hidden]
            edge_index: Edge indices [2, n_edges]
            train_disease_emb: Training set disease embeddings for sim decoder
            train_drug_scores: Known drug scores for training diseases
            
        Returns:
            Edge scores
        """
        src_emb = x_src[edge_index[0]]
        dst_emb = x_dst[edge_index[1]]
        
        if self.use_sim_decoder and train_disease_emb is not None:
            return self._sim_forward(src_emb, dst_emb, train_disease_emb, train_drug_scores)
        else:
            return self._dot_forward(src_emb, dst_emb)
    
    def _dot_forward(self, src_emb, dst_emb):
        """Standard dot-product scoring."""
        return (src_emb * dst_emb).sum(dim=-1)
    
    def _sim_forward(self, src_emb, dst_emb, train_disease_emb, train_drug_scores):
        """
        SimGNN-style similarity-weighted scoring for zero-shot.
        
        For new diseases, compute similarity to training diseases and 
        weight known drug-disease scores by similarity.
        """
        # Project for similarity computation
        query = self.sim_proj(dst_emb)  # [n_edges, hidden]
        keys = self.sim_proj(train_disease_emb)  # [n_train_diseases, hidden]
        
        # Compute similarity scores
        query_norm = F.normalize(query, dim=-1)
        keys_norm = F.normalize(keys, dim=-1)
        sim = torch.mm(query_norm, keys_norm.t()) / self.temp  # [n_edges, n_train]
        sim_weights = F.softmax(sim, dim=-1)
        
        # If we have training scores, use weighted combination
        if train_drug_scores is not None:
            # Aggregate known scores by similarity
            agg_scores = torch.mm(sim_weights, train_drug_scores)
            return agg_scores.squeeze(-1)
        else:
            # Fallback to dot product
            return self._dot_forward(src_emb, dst_emb)
    
    def compute_disease_similarity(self, disease_emb):
        """Compute disease-disease similarity matrix."""
        if not self.use_sim_decoder:
            return None
        
        proj = self.sim_proj(disease_emb)
        proj_norm = F.normalize(proj, dim=-1)
        return torch.mm(proj_norm, proj_norm.t())

