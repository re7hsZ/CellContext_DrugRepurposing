import torch
import torch.nn.functional as F
import os
import time
from src.evaluation.metrics import calculate_metrics, calculate_mrr

class Trainer:
    def __init__(self, model, predictor, optimizer, config, device, logger):
        self.model = model
        self.predictor = predictor
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.logger = logger
        
    def train_epoch(self, data):
        self.model.train()
        self.predictor.train()
        
        data = data.to(self.device)
        self.optimizer.zero_grad()
        
        # Encode
        z_dict = self.model(data.edge_index_dict)
        
        # Decode / Predict
        edge_label_index = data['drug', 'indication', 'disease'].edge_label_index
        edge_label = data['drug', 'indication', 'disease'].edge_label
        
        x_drug = z_dict['drug']
        x_disease = z_dict['disease']
        
        scores = self.predictor(x_drug, x_disease, edge_label_index)
        
        # Loss
        loss = F.binary_cross_entropy_with_logits(scores, edge_label)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data):
        self.model.eval()
        self.predictor.eval()
        data = data.to(self.device)
        
        z_dict = self.model(data.edge_index_dict)
        
        edge_label_index = data['drug', 'indication', 'disease'].edge_label_index
        edge_label = data['drug', 'indication', 'disease'].edge_label
        
        x_drug = z_dict['drug']
        x_disease = z_dict['disease']
        
        scores = self.predictor(x_drug, x_disease, edge_label_index)
        
        loss = F.binary_cross_entropy_with_logits(scores, edge_label).item()
        metrics = calculate_metrics(edge_label, scores)
        
        # MRR Heuristic
        pos_mask = edge_label == 1
        pos_scores = scores[pos_mask]
        neg_scores = scores[~pos_mask]
        
        if len(pos_scores) > 0 and len(neg_scores) > 0:
            metrics['MRR'] = calculate_mrr(pos_scores, neg_scores)
        else:
            metrics['MRR'] = 0.0
            
        return metrics, loss

    def fit(self, train_data, val_data):
        self.logger.info("Starting training loop...")
        best_auroc = 0.0
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_data)
            epoch_time = time.time() - start_time
            
            if epoch % 10 == 0:
                val_metrics, val_loss = self.evaluate(val_data)
                self.logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                                 f"AUROC: {val_metrics['auROC']:.4f} | AP: {val_metrics['auPRC']:.4f} | MRR: {val_metrics['MRR']:.4f}")
                
                # Checkpoint
                if val_metrics['auROC'] > best_auroc:
                    best_auroc = val_metrics['auROC']
                    self.save_checkpoint(epoch, val_metrics)
                    
    def save_checkpoint(self, epoch, metrics):
        save_dir = os.path.join(self.config['paths']['results_dir'], 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        path = os.path.join(save_dir, 'best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics
        }, path)
