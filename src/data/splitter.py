import torch
import numpy as np
from copy import deepcopy
from collections import Counter
from torch_geometric.transforms import RandomLinkSplit


def get_link_split(data, val_ratio=0.1, test_ratio=0.1, strategy='random', **kwargs):
    """Split graph for link prediction."""
    if strategy == 'random':
        return _random_split(data, val_ratio, test_ratio)
    elif strategy == 'disease':
        return _disease_split(data, val_ratio, test_ratio, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _random_split(data, val_ratio, test_ratio):
    """Standard random link split."""
    target = _find_target_edge(data)
    rev = _find_reverse_edge(data, target)
    
    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=1.0,
        edge_types=[target],
        rev_edge_types=[rev] if rev else None,
        add_negative_train_samples=True
    )
    return transform(data)


def _disease_split(data, val_ratio, test_ratio, min_edges_per_disease=2):
    """Zero-shot split by disease."""
    target = _find_target_edge(data)
    edge_index = data[target].edge_index
    
    drug_ids = edge_index[0].numpy()
    disease_ids = edge_index[1].numpy()
    
    counts = Counter(disease_ids)
    valid = [d for d, c in counts.items() if c >= min_edges_per_disease]
    
    if len(valid) < 3:
        return _random_split(data, val_ratio, test_ratio)
    
    np.random.shuffle(valid)
    n_test = max(1, int(len(valid) * test_ratio))
    n_val = max(1, int(len(valid) * val_ratio))
    
    test_diseases = set(valid[:n_test])
    val_diseases = set(valid[n_test:n_test + n_val])
    train_diseases = set(valid[n_test + n_val:]) | (set(disease_ids) - set(valid))
    
    train_mask = np.array([d in train_diseases for d in disease_ids])
    val_mask = np.array([d in val_diseases for d in disease_ids])
    test_mask = np.array([d in test_diseases for d in disease_ids])
    
    return _create_split_data(data, target, edge_index, train_mask, val_mask, test_mask)


def _find_target_edge(data):
    """Find drug-disease edge type."""
    for et in data.edge_types:
        if et[0] == 'drug' and et[2] == 'disease':
            return et
    raise ValueError("Target edge type not found")


def _find_reverse_edge(data, target):
    """Find reverse edge type."""
    for et in data.edge_types:
        if et[0] == target[2] and et[2] == target[0]:
            return et
    return None


def _create_split_data(data, target, edge_index, train_mask, val_mask, test_mask):
    """Create train/val/test data objects."""
    train_data = deepcopy(data)
    val_data = deepcopy(data)
    test_data = deepcopy(data)
    
    train_edges = edge_index[:, train_mask]
    val_edges = edge_index[:, val_mask]
    test_edges = edge_index[:, test_mask]
    
    train_data[target].edge_index = train_edges
    val_data[target].edge_index = train_edges
    test_data[target].edge_index = train_edges
    
    rev = _find_reverse_edge(data, target)
    if rev:
        train_data[rev].edge_index = train_edges[[1, 0]]
        val_data[rev].edge_index = train_edges[[1, 0]]
        test_data[rev].edge_index = train_edges[[1, 0]]
    
    n_drug = data['drug'].num_nodes
    n_disease = data['disease'].num_nodes
    
    def sample_neg(pos, n):
        pos_set = set(zip(pos[0].numpy(), pos[1].numpy()))
        neg = []
        while len(neg) < n:
            d, s = np.random.randint(n_drug), np.random.randint(n_disease)
            if (d, s) not in pos_set:
                neg.append([d, s])
        return torch.tensor(neg, dtype=torch.long).t()
    
    for split_data, edges in [(train_data, train_edges), 
                               (val_data, val_edges), 
                               (test_data, test_edges)]:
        neg = sample_neg(edges, edges.shape[1])
        split_data[target].edge_label_index = torch.cat([edges, neg], dim=1)
        split_data[target].edge_label = torch.cat([
            torch.ones(edges.shape[1]),
            torch.zeros(neg.shape[1])
        ])
    
    return train_data, val_data, test_data
