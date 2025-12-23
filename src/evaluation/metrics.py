from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import torch
import numpy as np

def calculate_metrics(y_true, y_scores, threshold=0.5):
    """
    y_true: true labels (0 or 1)
    y_scores: predicted logits or probabilities
    threshold: threshold for F1 score (prob > threshold -> 1)
    """
    # Convert to numpy
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.detach().cpu().numpy()
        
    # Apply sigmoid to scores if they are logits (metrics often handle scores directly, but F1 needs classes)
    # Assuming y_scores are logits here for flexibility, lets apply sigmoid
    # But usually AP/AUC don't span sigmoid.
    # We will assume y_scores are raw logits.
    probs = 1 / (1 + np.exp(-y_scores))
    preds = (probs > threshold).astype(int)
    
    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = 0.5 # One class present
        
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = 0.0
        
    f1 = f1_score(y_true, preds)
    
    return {
        'auROC': auroc,
        'auPRC': auprc,
        'F1': f1
    }

def calculate_mrr(pos_scores, neg_scores):
    """
    Calculate MRR for a batch.
    pos_scores: Scores of positive edges [N]
    neg_scores: Scores of negative edges [N] (1-to-1 paired) or [N, K]
    
    Simplified MRR:
    Rank positive item against all negative items.
    rank = 1 + count(neg_score > pos_score)
    reciprocal_rank = 1 / rank
    """
    # Simply 1-vs-1 comparison if lengths match and we treat them as pairs?
    # Usually we want rank of pos in a larger set.
    # For now, let's implement a naive version:
    # Concatenate pos[i] with ALL negatives and rank? Expensive.
    # Let's concatenate pos[i] with neg[i] (if paired).
    # Better: rank = 1 + sum(neg_scores > pos_score) (Vectorized)
    
    if torch.is_tensor(pos_scores):
        pos_scores = pos_scores.detach().cpu()
    if torch.is_tensor(neg_scores):
        neg_scores = neg_scores.detach().cpu()
        
    # We assume 'neg_scores' is a pool of negatives.
    # If neg_scores is large, we might sample.
    
    # Expand dims for broadcasting
    # pos: [N, 1], neg: [1, M]
    # sum(neg > pos) -> [N]
    
    # If neg_scores is same size as pos_scores (1:1), this is just checking if pos > neg. 
    # That's not a real rank.
    # Training usually provides 1:1 negatives.
    # Evaluation usually provides 1:1 too by RandomLinkSplit.
    # True MRR requires 1:Many.
    # We will skip MRR for the training loop log and only calculate it if we have a way to get many negatives.
    # Or we treat the entire batch of negatives as the candidate set.
    
    # "Batch MRR":
    # For each positive i, compare against ALL negatives in the batch.
    # pos_scores: [N]
    # neg_scores: [N]
    
    p = pos_scores.view(-1, 1) # [N, 1]
    n = neg_scores.view(1, -1) # [1, N]
    
    # rank_i = 1 + count(n_j > p_i)
    ranks = 1 + (n > p).sum(dim=1)
    
    mrr = (1.0 / ranks).mean().item()
    return mrr

def recall_at_k(pos_scores, neg_scores, k=10):
    """
    Calculate Recall@K.
    For each positive, check if it ranks in top K among all negatives.
    
    pos_scores: [N] positive scores
    neg_scores: [M] negative scores (treated as candidate pool)
    k: Top K threshold
    """
    if torch.is_tensor(pos_scores):
        pos_scores = pos_scores.detach().cpu()
    if torch.is_tensor(neg_scores):
        neg_scores = neg_scores.detach().cpu()
        
    # For each positive, compute rank
    p = pos_scores.view(-1, 1)  # [N, 1]
    n = neg_scores.view(1, -1)  # [1, M]
    
    # rank_i = 1 + count(n_j > p_i)
    ranks = 1 + (n > p).sum(dim=1)
    
    # Recall@K = fraction of positives ranked in top K
    hits = (ranks <= k).float()
    recall = hits.mean().item()
    return recall
