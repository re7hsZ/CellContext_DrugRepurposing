import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.data.dataset_loader import DatasetLoader
from src.data.splitter import get_link_split
from src.models.gcn_encoder import HeteroGCN
from src.models.link_predictor import LinkPredictor
from src.evaluation.metrics import calculate_metrics, calculate_mrr

def train(model, predictor, loader, optimizer, device):
    model.train()
    predictor.train()
    
    total_loss = 0
    # loader serves batches, but here 'loader' is just the data object usually for full-batch GCN.
    # If graph is small enough, full batch is fine. 
    # RandomLinkSplit returns Data objects, we can use them directly.
    
    data = loader # It's a single HeteroData object
    data = data.to(device)
    
    optimizer.zero_grad()
    
    # 1. Encode -> Get Node Embeddings
    z_dict = model(data.edge_index_dict)
    
    # 2. Decode -> Get Edge Scores
    # We predict on 'edge_label_index'
    edge_label_index = data['drug', 'indication', 'disease'].edge_label_index
    edge_label = data['drug', 'indication', 'disease'].edge_label
    
    # Embeddings
    x_drug = z_dict['drug']
    x_disease = z_dict['disease']
    
    # Predict
    scores = predictor(x_drug, x_disease, edge_label_index)
    
    # Loss
    loss = F.binary_cross_entropy_with_logits(scores, edge_label)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

@torch.no_grad()
def evaluate(model, predictor, data, device):
    model.eval()
    predictor.eval()
    data = data.to(device)
    
    z_dict = model(data.edge_index_dict)
    
    edge_label_index = data['drug', 'indication', 'disease'].edge_label_index
    edge_label = data['drug', 'indication', 'disease'].edge_label
    
    x_drug = z_dict['drug']
    x_disease = z_dict['disease']
    
    scores = predictor(x_drug, x_disease, edge_label_index)
    
    # Metrics
    metrics = calculate_metrics(edge_label, scores)
    
    # Calculate MRR (Batch-wise heuristic)
    # Filter pos/neg scores
    pos_mask = edge_label == 1
    pos_scores = scores[pos_mask]
    neg_scores = scores[~pos_mask]
    
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        metrics['MRR'] = calculate_mrr(pos_scores, neg_scores)
    else:
        metrics['MRR'] = 0.0
        
    return metrics, F.binary_cross_entropy_with_logits(scores, edge_label).item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', type=str, default='microglial_cell.txt', help='Cell type PPI file')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # 1. Load Data
    loader = DatasetLoader()
    print(f"Constructing graph for {args.cell_type}...")
    data = loader.create_hetero_data(args.cell_type)
    
    # 2. Split
    print("Splitting data...")
    train_data, val_data, test_data = get_link_split(data)
    
    # 3. Model
    # Need number of nodes for embedding
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    
    model = HeteroGCN(data.metadata(), hidden_channels=args.hidden_dim, num_nodes_dict=num_nodes_dict).to(args.device)
    predictor = LinkPredictor().to(args.device)
    
    optimizer = torch.optim.Adam(list(model.parameters()) + list(predictor.parameters()), lr=args.lr)
    
    # 4. Loop
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        loss = train(model, predictor, train_data, optimizer, args.device)
        
        if epoch % 10 == 0:
            val_metrics, val_loss = evaluate(model, predictor, val_data, args.device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"AUROC: {val_metrics['auROC']:.4f} | AUPRC: {val_metrics['auPRC']:.4f} | MRR: {val_metrics['MRR']:.4f}")
            
    # 5. Test
    print("Testing...")
    test_metrics, test_loss = evaluate(model, predictor, test_data, args.device)
    print(f"Test Results | Loss: {test_loss:.4f} | "
          f"AUROC: {test_metrics['auROC']:.4f} | AUPRC: {test_metrics['auPRC']:.4f} | MRR: {test_metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()
