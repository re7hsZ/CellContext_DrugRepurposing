import os
import torch


class DatasetLoader:
    """Load pre-built graph objects from disk."""
    
    def __init__(self, processed_dir='data/processed'):
        self.processed_dir = processed_dir

    def load(self, graph_name):
        """Load graph by name."""
        path = os.path.join(self.processed_dir, f'{graph_name}_graph.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph not found: {path}")
        return torch.load(path, weights_only=False)
