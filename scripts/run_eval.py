"""
Evaluation script: loads a trained model and evaluates on test set.
"""
import argparse
import sys
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.utils.helpers import load_config, get_device
from src.utils.logger import setup_logger
from src.data.dataset_loader import DatasetLoader
from src.data.splitter import get_link_split
from src.models.gcn_encoder import HeteroGCN
from src.models.link_predictor import LinkPredictor
from src.evaluation.metrics import calculate_metrics, calculate_mrr
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.yaml')
    parser.add_argument('--checkpoint', type=str, default='results/checkpoints/best_model.pth')
    args = parser.parse_args()
    
    config = load_config(args.config)
    logger = setup_logger('eval', os.path.join(config['paths']['results_dir'], 'logs'))
    device = get_device(config['training']['device'])
    
    # Load Data
    loader = DatasetLoader(config['paths']['processed_dir'])
    cell_type_file = config['data']['cell_type']
    graph_name = os.path.splitext(cell_type_file)[0]
    data = loader.load(graph_name)
    
    # Split (to get test set)
    _, _, test_data = get_link_split(data, 
                                     val_ratio=config['split']['val_ratio'], 
                                     test_ratio=config['split']['test_ratio'])
    
    # Model
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroGCN(data.metadata(), 
                      hidden_channels=config['model']['hidden_channels'], 
                      num_layers=config['model']['num_layers'], 
                      num_nodes_dict=num_nodes_dict).to(device)
    predictor = LinkPredictor().to(device)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    predictor.load_state_dict(checkpoint['predictor_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    model.eval()
    predictor.eval()
    test_data = test_data.to(device)
    
    with torch.no_grad():
        z_dict = model(test_data.edge_index_dict)
        edge_label_index = test_data['drug', 'indication', 'disease'].edge_label_index
        edge_label = test_data['drug', 'indication', 'disease'].edge_label
        
        x_drug = z_dict['drug']
        x_disease = z_dict['disease']
        scores = predictor(x_drug, x_disease, edge_label_index)
        
        loss = F.binary_cross_entropy_with_logits(scores, edge_label).item()
        metrics = calculate_metrics(edge_label, scores)
        
        pos_mask = edge_label == 1
        pos_scores = scores[pos_mask]
        neg_scores = scores[~pos_mask]
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            metrics['MRR'] = calculate_mrr(pos_scores, neg_scores)
        else:
            metrics['MRR'] = 0.0
            
    logger.info(f"=== Test Evaluation ===")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"AUROC: {metrics['auROC']:.4f}")
    logger.info(f"AUPRC: {metrics['auPRC']:.4f}")
    logger.info(f"F1: {metrics['F1']:.4f}")
    logger.info(f"MRR: {metrics['MRR']:.4f}")

if __name__ == "__main__":
    main()
