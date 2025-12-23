import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed, get_device, setup_logger
from src.data import DatasetLoader, get_link_split
from src.models import HeteroGCN, LinkPredictor
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_base.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    log_dir = os.path.join(config['paths']['results_dir'], 'logs')
    logger = setup_logger('train', log_dir)
    
    set_seed(config['training']['seed'])
    device = get_device(config['training']['device'])
    logger.info(f"Device: {device}")
    
    # Load data
    loader = DatasetLoader(config['paths']['processed_dir'])
    
    cell_type = config['data']['cell_type']
    use_pinnacle = config['data'].get('use_pinnacle_features', True)
    
    graph_name = os.path.splitext(cell_type)[0]
    if use_pinnacle and cell_type != 'general':
        graph_name += '_pinnacle'
    
    try:
        data = loader.load(graph_name)
    except FileNotFoundError:
        logger.error(f"Graph not found. Run run_preprocess.py first.")
        return
    
    logger.info(f"Loaded: {graph_name}")
    
    # Split
    split_cfg = config.get('split', {})
    train_data, val_data, test_data = get_link_split(
        data,
        val_ratio=split_cfg.get('val_ratio', 0.1),
        test_ratio=split_cfg.get('test_ratio', 0.1),
        strategy=split_cfg.get('strategy', 'random'),
        min_edges_per_disease=split_cfg.get('min_edges_per_disease', 2)
    )
    
    # Verify negative sampling
    target = ('drug', 'indication', 'disease')
    train_labels = train_data[target].edge_label
    pos_ratio = train_labels.float().mean().item()
    logger.info(f"[Neg Sampling] Train: pos={int(train_labels.sum())}, neg={int((train_labels == 0).sum())}")
    
    if pos_ratio > 0.99:
        logger.error("No negative samples! Check splitter.py")
        return
    
    # Check PINNACLE feature usage
    gene_feat = data['gene'].x
    non_zero = (gene_feat.abs().sum(dim=1) > 0).sum().item()
    logger.info(f"[PINNACLE] Gene features: {non_zero}/{data['gene'].num_nodes} non-zero")
    
    # Model
    model_cfg = config['model']
    num_nodes = {nt: data[nt].num_nodes for nt in data.node_types}
    
    model = HeteroGCN(
        data.metadata(),
        hidden_channels=model_cfg.get('hidden_channels', 64),
        num_layers=model_cfg.get('num_layers', 2),
        num_nodes_dict=num_nodes
    ).to(device)
    
    predictor = LinkPredictor().to(device)
    
    # Dummy forward to initialize LazyLinear parameters BEFORE optimizer
    with torch.no_grad():
        dummy_x = {nt: train_data[nt].x.to(device) for nt in train_data.node_types}
        _ = model(dummy_x, train_data.to(device).edge_index_dict)
    logger.info("[Init] LazyLinear initialized via dummy forward")
    
    # Optimizer - now LazyLinear parameters are properly initialized
    train_cfg = config['training']
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()),
        lr=train_cfg.get('learning_rate', 0.01),
        weight_decay=train_cfg.get('weight_decay', 0.0005)
    )
    
    # Train
    trainer = Trainer(model, predictor, optimizer, config, device, logger)
    trainer.fit(train_data, val_data)
    
    # Test
    logger.info("Final test evaluation")
    metrics, loss = trainer.evaluate(test_data)
    logger.info(f"Test AUROC: {metrics['auROC']:.4f} | AP: {metrics['auPRC']:.4f} | MRR: {metrics['MRR']:.4f}")


if __name__ == '__main__':
    main()
