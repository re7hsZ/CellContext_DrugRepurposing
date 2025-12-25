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


def calculate_recall_at_k(pos_scores, neg_scores, k=50):
    """Calculate Recall@K metric."""
    if torch.is_tensor(pos_scores):
        pos_scores = pos_scores.detach().cpu().numpy()
    if torch.is_tensor(neg_scores):
        neg_scores = neg_scores.detach().cpu().numpy()
    
    n_pos = len(pos_scores)
    if n_pos == 0:
        return 0.0
    
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(len(neg_scores))])
    
    sorted_idx = np.argsort(-all_scores)
    top_k_labels = all_labels[sorted_idx[:k]]
    
    return top_k_labels.sum() / min(n_pos, k)


def calculate_ns_recall(scores, labels, disease_ids, k=50):
    """
    Calculate Normalized Sensitivity Recall (NS-Recall) from TxGNN.
    
    This metric addresses popularity bias by normalizing recall per disease,
    giving equal weight to rare and common diseases.
    
    Args:
        scores: Prediction scores for all drug-disease pairs
        labels: Ground truth labels (1=positive, 0=negative)
        disease_ids: Disease ID for each prediction
        k: Top-k threshold for recall calculation
        
    Returns:
        NS-Recall value (mean of per-disease normalized recalls)
    """
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    if torch.is_tensor(disease_ids):
        disease_ids = disease_ids.detach().cpu().numpy()
    
    unique_diseases = np.unique(disease_ids)
    disease_recalls = []
    
    for disease in unique_diseases:
        mask = disease_ids == disease
        d_scores = scores[mask]
        d_labels = labels[mask]
        
        n_pos = d_labels.sum()
        if n_pos == 0:
            continue
        
        # Sort by score descending
        sorted_idx = np.argsort(-d_scores)
        sorted_labels = d_labels[sorted_idx]
        
        # Calculate recall at k for this disease
        top_k = sorted_labels[:k]
        recall = top_k.sum() / min(n_pos, k)
        disease_recalls.append(recall)
    
    if len(disease_recalls) == 0:
        return 0.0
    
    return np.mean(disease_recalls)

