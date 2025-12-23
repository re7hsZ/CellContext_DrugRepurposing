import os
import torch
import torch.nn.functional as F
from src.evaluation.metrics import calculate_metrics, calculate_mrr


class Trainer:
    """Model trainer with validation and checkpointing."""
    
    def __init__(self, model, predictor, optimizer, config, device, logger):
        self.model = model
        self.predictor = predictor
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        self.target_edge = ('drug', 'indication', 'disease')

    def train_epoch(self, data):
        """Run one training epoch."""
        self.model.train()
        self.predictor.train()
        
        data = data.to(self.device)
        self.optimizer.zero_grad()
        
        z_dict = self.model(data.edge_index_dict)
        
        edge_label_index = data[self.target_edge].edge_label_index
        edge_label = data[self.target_edge].edge_label
        
        scores = self.predictor(z_dict['drug'], z_dict['disease'], edge_label_index)
        loss = F.binary_cross_entropy_with_logits(scores, edge_label)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        """Evaluate model on given data."""
        self.model.eval()
        self.predictor.eval()
        
        data = data.to(self.device)
        z_dict = self.model(data.edge_index_dict)
        
        edge_label_index = data[self.target_edge].edge_label_index
        edge_label = data[self.target_edge].edge_label
        
        scores = self.predictor(z_dict['drug'], z_dict['disease'], edge_label_index)
        loss = F.binary_cross_entropy_with_logits(scores, edge_label).item()
        
        metrics = calculate_metrics(edge_label, scores)
        
        pos_mask = edge_label == 1
        if pos_mask.sum() > 0 and (~pos_mask).sum() > 0:
            metrics['MRR'] = calculate_mrr(scores[pos_mask], scores[~pos_mask])
        else:
            metrics['MRR'] = 0.0
        
        return metrics, loss

    def fit(self, train_data, val_data):
        """Run full training loop."""
        self.logger.info("Starting training")
        best_auroc = 0.0
        epochs = self.config['training']['epochs']
        
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_data)
            
            if epoch % 10 == 0:
                metrics, val_loss = self.evaluate(val_data)
                self.logger.info(
                    f"Epoch {epoch:03d} | Loss: {loss:.4f} | "
                    f"Val: {val_loss:.4f} | AUROC: {metrics['auROC']:.4f} | "
                    f"AP: {metrics['auPRC']:.4f} | MRR: {metrics['MRR']:.4f}"
                )
                
                if metrics['auROC'] > best_auroc:
                    best_auroc = metrics['auROC']
                    self._save_checkpoint(epoch, metrics)

    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint."""
        save_dir = os.path.join(self.config['paths']['results_dir'], 'checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'metrics': metrics
        }, os.path.join(save_dir, 'best_model.pth'))
