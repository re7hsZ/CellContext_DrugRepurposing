import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """Calculate classification metrics."""
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.detach().cpu().numpy()
    
    probs = 1 / (1 + np.exp(-y_scores))
    preds = (probs > threshold).astype(int)
    
    try:
        auroc = roc_auc_score(y_true, probs)
    except ValueError:
        auroc = 0.5
    
    try:
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        auprc = 0.0
    
    f1 = f1_score(y_true, preds)
    
    return {'auROC': auroc, 'auPRC': auprc, 'F1': f1}


def calculate_mrr(pos_scores, neg_scores):
    """Calculate mean reciprocal rank."""
    if torch.is_tensor(pos_scores):
        pos_scores = pos_scores.detach().cpu()
    if torch.is_tensor(neg_scores):
        neg_scores = neg_scores.detach().cpu()
    
    pos = pos_scores.view(-1, 1)
    neg = neg_scores.view(1, -1)
    ranks = 1 + (neg > pos).sum(dim=1)
    
    return (1.0 / ranks).mean().item()
