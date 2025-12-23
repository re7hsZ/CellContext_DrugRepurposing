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
    
    # Optimizer
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
    logger.info(f"Test AUROC: {metrics['auROC']:.4f} | AP: {metrics['auPRC']:.4f}")


if __name__ == '__main__':
    main()
