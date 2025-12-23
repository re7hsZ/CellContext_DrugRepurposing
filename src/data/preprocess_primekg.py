import os
import pandas as pd

RAW_PATH = 'data/raw/primekg/kg.csv'
OUTPUT_DIR = 'data/processed'


def process_primekg():
    """Process raw PrimeKG data into separate edge files."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading PrimeKG from {RAW_PATH}...")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    
    print("Relation types:", df['relation'].unique())
    print("Node types:", df['x_type'].unique())
    
    # Extract node mapping
    print("Extracting node information...")
    
    nodes_x = df[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].rename(columns={
        'x_index': 'original_index', 
        'x_id': 'node_id', 
        'x_type': 'node_type', 
        'x_name': 'node_name', 
        'x_source': 'source'
    })
    nodes_y = df[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].rename(columns={
        'y_index': 'original_index', 
        'y_id': 'node_id', 
        'y_type': 'node_type', 
        'y_name': 'node_name', 
        'y_source': 'source'
    })
    
    all_nodes = pd.concat([nodes_x, nodes_y]).drop_duplicates(
        subset=['node_id', 'node_type']
    ).reset_index(drop=True)
    all_nodes['node_idx'] = all_nodes.index
    
    all_nodes.to_csv(os.path.join(OUTPUT_DIR, 'node_mapping.csv'), index=False)
    print(f"Saved {len(all_nodes)} nodes to node_mapping.csv")
    
    # Extract edge types
    print("Splitting edges...")
    
    # Drug-Gene edges
    edge_drug_gene = df[
        ((df['x_type'] == 'drug') & (df['y_type'] == 'gene/protein')) |
        ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'drug'))
    ].copy()
    edge_drug_gene.to_csv(os.path.join(OUTPUT_DIR, 'edges_drug_gene.csv'), index=False)
    print(f"Drug-Gene edges: {len(edge_drug_gene)}")
    
    # Gene-Disease edges
    edge_gene_disease = df[
        ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'disease')) |
        ((df['x_type'] == 'disease') & (df['y_type'] == 'gene/protein'))
    ].copy()
    edge_gene_disease.to_csv(os.path.join(OUTPUT_DIR, 'edges_gene_disease.csv'), index=False)
    print(f"Gene-Disease edges: {len(edge_gene_disease)}")
    
    # Drug-Disease edges (labels for prediction)
    edge_drug_disease = df[
        ((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) |
        ((df['x_type'] == 'disease') & (df['y_type'] == 'drug'))
    ].copy()
    
    if 'indication' in df['relation'].unique():
        edge_drug_disease = edge_drug_disease[edge_drug_disease['relation'] == 'indication']
    
    edge_drug_disease.to_csv(os.path.join(OUTPUT_DIR, 'edges_drug_disease_gold.csv'), index=False)
    print(f"Drug-Disease labels: {len(edge_drug_disease)}")
    
    # General PPI (to be replaced by PINNACLE)
    edge_ppi = df[df['relation'] == 'protein_protein'].copy()
    edge_ppi.to_csv(os.path.join(OUTPUT_DIR, 'edges_ppi_general.csv'), index=False)
    print(f"General PPI edges: {len(edge_ppi)}")
    
    print("Done.")


if __name__ == "__main__":
    process_primekg()