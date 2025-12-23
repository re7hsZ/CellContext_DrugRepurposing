import torch
from torch_geometric.transforms import RandomLinkSplit

def get_link_split(data, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the HeteroData into train, val, and test sets for link prediction.
    Target edge is ('drug', 'indication', 'disease').
    """
    # Identify target edge type
    target_edge = ('drug', 'indication', 'disease')
    if target_edge not in data.edge_types:
        # Fallback search
        found = False
        for et in data.edge_types:
            if et[0] == 'drug' and et[2] == 'disease':
                target_edge = et
                found = True
                break
        if not found:
             # Maybe strict naming was different?
             # Let's print available to debug if error
             raise ValueError(f"Target edge ('drug', 'indication', 'disease') not found. Available: {data.edge_types}")

    # Identify reverse edge type (to avoid data leakage by removing reverse of validation/test edges from message passing)
    rev_edge = None
    # Heuristic: look for (disease, *, drug)
    for et in data.edge_types:
        if et[0] == 'disease' and et[2] == 'drug':
            rev_edge = et 
            break
            
    print(f"Splitting data with Target: {target_edge}, Reverse: {rev_edge}")

    transform = RandomLinkSplit(
        is_undirected=False,
        num_val=val_ratio,
        num_test=test_ratio,
        neg_sampling_ratio=1.0, # 1 negative for each positive
        edge_types=[target_edge],
        rev_edge_types=[rev_edge] if rev_edge else None, 
        add_negative_train_samples=True 
    )
    
    train_data, val_data, test_data = transform(data)
    
    return train_data, val_data, test_data
