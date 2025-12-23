import torch
import os

class DatasetLoader:
    def __init__(self, processed_dir='data/processed'):
        self.processed_dir = processed_dir

    def load(self, graph_name):
        """
        Loads a pre-processed HeteroData object from disk.
        graph_name: e.g. 'microglial_cell' or 'general'
        """
        path = os.path.join(self.processed_dir, f'{graph_name}_graph.pt')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Graph file not found: {path}. Please run run_preprocess.py first.")
            
        print(f"Loading graph from {path}...")
        # PyG Data objects are not just weights, so we need weights_only=False (or safe globals)
        # Explicitly convert path to string to be safe
        data = torch.load(str(path), weights_only=False)
        return data
