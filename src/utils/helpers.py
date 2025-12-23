import yaml
import torch
import numpy as np
import random


def load_config(path):
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_str='auto'):
    """Get torch device from string specification."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)
