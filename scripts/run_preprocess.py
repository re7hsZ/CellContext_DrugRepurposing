import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import GraphBuilder
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config_base.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    cell_type = config['data']['cell_type']
    use_pinnacle = config['data'].get('use_pinnacle_features', True)
    
    graph_name = os.path.splitext(cell_type)[0]
    if use_pinnacle and cell_type != 'general':
        graph_name += '_pinnacle'
    
    print(f"Building graph: {graph_name}")
    
    builder = GraphBuilder(
        data_dir=config['paths']['processed_dir'],
        raw_dir=config['paths']['raw_dir']
    )
    
    data = builder.build(cell_type, use_pinnacle=use_pinnacle)
    builder.save(data, graph_name)
    
    print(f"Done. Nodes: {data.node_types}, Edges: {data.edge_types}")


if __name__ == '__main__':
    main()
