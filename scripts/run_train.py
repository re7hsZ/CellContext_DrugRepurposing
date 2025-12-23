import argparse
import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.utils.helpers import load_config, set_seed, get_device
from src.utils.logger import setup_logger
from src.data.dataset_loader import DatasetLoader
from src.data.splitter import get_link_split
from src.models.gcn_encoder import HeteroGCN
from src.models.link_predictor import LinkPredictor
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.yaml')
    args = parser.parse_args()
    
    # 1. Config & Setup
    config = load_config(args.config)
    logger = setup_logger('train', os.path.join(config['paths']['results_dir'], 'logs'))
    
    set_seed(config['training']['seed'])
    device = get_device(config['training']['device'])
    logger.info(f"Device: {device}")
    
    # 2. Load Data
    loader = DatasetLoader(config['paths']['processed_dir'])
    cell_type_file = config['data']['cell_type']
    graph_name = os.path.splitext(cell_type_file)[0]
    
    try:
        data = loader.load(graph_name)
    except FileNotFoundError:
        logger.error(f"Graph {graph_name} not found. Run scripts/run_preprocess.py first.")
        return

    # 3. Split
    train_data, val_data, test_data = get_link_split(data, 
                                                     val_ratio=config['split']['val_ratio'], 
                                                     test_ratio=config['split']['test_ratio'])
    
    # 4. Model
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroGCN(data.metadata(), 
                      hidden_channels=config['model']['hidden_channels'], 
                      num_layers=config['model']['num_layers'], 
                      num_nodes_dict=num_nodes_dict).to(device)
                      
    predictor = LinkPredictor().to(device)
    
    # 5. Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(predictor.parameters()), 
        lr=config['training']['learning_rate'], 
        weight_decay=config['training']['weight_decay']
    )
    
    # 6. Trainer
    trainer = Trainer(model, predictor, optimizer, config, device, logger)
    trainer.fit(train_data, val_data)
    
    # 7. Final Test
    logger.info("Running final test...")
    test_metrics, test_loss = trainer.evaluate(test_data)
    logger.info(f"Test Results | Loss: {test_loss:.4f} | AUROC: {test_metrics['auROC']:.4f}")

if __name__ == "__main__":
    main()
