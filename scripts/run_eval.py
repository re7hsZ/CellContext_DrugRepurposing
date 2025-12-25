import os
import sys
import argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, get_device, setup_logger, set_seed
from src.data import DatasetLoader, get_link_split
from src.models import HeteroGCN, LinkPredictor
from src.evaluation import calculate_metrics, calculate_mrr, calculate_recall_at_k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_base.yaml')
    parser.add_argument('--checkpoint', default=None)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    set_seed(config['training']['seed'])
    
    log_dir = os.path.join(config['paths']['results_dir'], 'logs')
    logger = setup_logger('eval', log_dir)
    
    device = get_device(config['training']['device'])
    
    loader = DatasetLoader(config['paths']['processed_dir'])
    
    cell_type = config['data']['cell_type']
    use_pinnacle = config['data'].get('use_pinnacle_features', True)
    use_text_embeddings = config['data'].get('use_text_embeddings', False)
    
    graph_name = os.path.splitext(cell_type)[0]
    if use_pinnacle and cell_type != 'general':
        graph_name += '_pinnacle'
    if use_text_embeddings:
        graph_name += '_textembed'
    
    data = loader.load(graph_name)
    
    split_cfg = config.get('split', {})
    _, _, test_data = get_link_split(
        data,
        val_ratio=split_cfg.get('val_ratio', 0.1),
        test_ratio=split_cfg.get('test_ratio', 0.1),
        strategy=split_cfg.get('strategy', 'random')
    )
    
    model_cfg = config['model']
    num_nodes = {nt: data[nt].num_nodes for nt in data.node_types}
    
    model = HeteroGCN(
        data.metadata(),
        hidden_channels=model_cfg.get('hidden_channels', 64),
        num_layers=model_cfg.get('num_layers', 2),
        num_nodes_dict=num_nodes
    ).to(device)
    
    with torch.no_grad():
        dummy_x = {nt: test_data[nt].x.to(device) for nt in test_data.node_types}
        _ = model(dummy_x, test_data.to(device).edge_index_dict)
    
    # Configure predictor with optional similarity decoder for zero-shot
    use_sim = model_cfg.get('use_sim_decoder', False)
    predictor = LinkPredictor(
        hidden_channels=model_cfg.get('hidden_channels', 64) if use_sim else None,
        use_sim_decoder=use_sim
    ).to(device)
    
    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        cell_name = os.path.splitext(cell_type)[0]
        ckpt_path = os.path.join(
            config['paths']['results_dir'], 
            'checkpoints', 
            f'best_model_{cell_name}.pth'
        )
    
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    predictor.load_state_dict(ckpt['predictor_state_dict'])
    logger.info(f"Loaded checkpoint: {ckpt_path} (epoch {ckpt['epoch']})")
    
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
            metrics['Recall@50'] = calculate_recall_at_k(scores[pos_mask], scores[~pos_mask], k=50)
        else:
            metrics['MRR'] = 0.0
            metrics['Recall@50'] = 0.0
    
    logger.info(f"Test Loss: {loss:.4f}")
    logger.info(f"AUROC: {metrics['auROC']:.4f}")
    logger.info(f"AUPRC: {metrics['auPRC']:.4f}")
    logger.info(f"MRR: {metrics['MRR']:.4f}")
    logger.info(f"Recall@50: {metrics['Recall@50']:.4f}")


if __name__ == '__main__':
    main()
