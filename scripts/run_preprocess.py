import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from src.data.graph_builder import GraphBuilder
from src.utils.helpers import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_base.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    cell_type_file = config['data']['cell_type']
    graph_name = os.path.splitext(cell_type_file)[0] # e.g. microglial_cell
    
    print(f"Running preprocessing for: {graph_name}")
    
    builder = GraphBuilder()
    data = builder.build_graph(cell_type_file)
    
    builder.save_graph(data, graph_name)
    print("Preprocessing Done.")

if __name__ == "__main__":
    main()
