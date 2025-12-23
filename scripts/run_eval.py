import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, setup_logger, set_seed
from src.data import DatasetLoader, get_link_split
from src.models import HeteroGCN, LinkPredictor
from src.evaluation import calculate_metrics, calculate_mrr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_base.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Set seed to ensure same data split as training
    set_seed(config['training']['seed'])
    
    log_dir = os.path.join(config['paths']['results_dir'], 'logs')
    logger = setup_logger('eval', log_dir)
    
    device = get_device(config['training']['device'])
    
    # Load data
    loader = DatasetLoader(config['paths']['processed_dir'])
    
    cell_type = config['data']['cell_type']
    use_pinnacle = config['data'].get('use_pinnacle_features', True)
    
    graph_name = os.path.splitext(cell_type)[0]
    if use_pinnacle and cell_type != 'general':
        graph_name += '_pinnacle'
    
    data = loader.load(graph_name)
    
    split_cfg = config.get('split', {})
    _, _, test_data = get_link_split(
        data,
        val_ratio=split_cfg.get('val_ratio', 0.1),
        test_ratio=split_cfg.get('test_ratio', 0.1),
        strategy=split_cfg.get('strategy', 'random')
    )
    
    # Model
    model_cfg = config['model']
    num_nodes = {nt: data[nt].num_nodes for nt in data.node_types}
    
    model = HeteroGCN(
        data.metadata(),
        hidden_channels=model_cfg.get('hidden_channels', 64),
        num_layers=model_cfg.get('num_layers', 2),
        num_nodes_dict=num_nodes
    ).to(device)
    
    # Dummy forward to initialize LazyLinear before loading weights
    with torch.no_grad():
        dummy_x = {nt: test_data[nt].x.to(device) for nt in test_data.node_types}
        _ = model(dummy_x, test_data.to(device).edge_index_dict)
    
    predictor = LinkPredictor().to(device)
    
    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        cell_name = os.path.splitext(cell_type)[0]
        ckpt_path = os.path.join(
            config['paths']['results_dir'], 
            'checkpoints', 
            f'best_model_{cell_name}.pth'
        )
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
    
    # Evaluate
    model.eval()
    predictor.eval()
    
    test_data = test_data.to(device)
    target = ('drug', 'indication', 'disease')
    
    with torch.no_grad():
        x_dict = {nt: test_data[nt].x for nt in test_data.node_types}
        z_dict = model(x_dict, test_data.edge_index_dict)
        
        edge_idx = test_data[target].edge_label_index
        edge_label = test_data[target].edge_label
        
        scores = predictor(z_dict['drug'], z_dict['disease'], edge_idx)
        loss = F.binary_cross_entropy_with_logits(scores, edge_label.float()).item()
        
        metrics = calculate_metrics(edge_label, scores)
        
        pos_mask = edge_label == 1
        if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
            metrics['MRR'] = calculate_mrr(scores[pos_mask], scores[~pos_mask])
        else:
            metrics['MRR'] = 0.0
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"AUROC: {metrics['auROC']:.4f}")
    logger.info(f"AUPRC: {metrics['auPRC']:.4f}")
    logger.info(f"MRR: {metrics['MRR']:.4f}")


if __name__ == '__main__':
    main()
