import os
import torch
import pandas as pd
from torch_geometric.data import HeteroData

class GraphBuilder:
    def __init__(self, data_dir='data/processed', raw_dir='data/raw'):
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.node_mapping = None
        self.global_to_local_type = {} 
        self.original_to_global = {} 
        self.gene_symbol_to_global = {} 
        self.type_counts = {}

    def load_node_mapping(self):
        mapping_path = os.path.join(self.data_dir, 'node_mapping.csv')
        print(f"Loading node mapping from {mapping_path}...")
        self.node_mapping = pd.read_csv(mapping_path)
        
        self.original_to_global = dict(zip(self.node_mapping['original_index'], self.node_mapping['node_idx']))
        
        grouped = self.node_mapping.groupby('node_type')
        for node_type, group in grouped:
            clean_type = self.sanitize_type(node_type)
            if clean_type == 'other': continue
            sorted_group = group.sort_values('node_idx')
            local_map = {global_idx: i for i, global_idx in enumerate(sorted_group['node_idx'])}
            self.type_counts[clean_type] = len(sorted_group)
            for g_idx, l_idx in local_map.items():
                self.global_to_local_type[g_idx] = (clean_type, l_idx)

        genes = self.node_mapping[self.node_mapping['node_type'] == 'gene/protein']
        self.gene_symbol_to_global = dict(zip(genes['node_name'], genes['node_idx']))
        print(f"Node mapping loaded. Counts: {self.type_counts}")

    def sanitize_type(self, t):
        if t == 'gene/protein': return 'gene'
        if t == 'drug': return 'drug'
        if t == 'disease': return 'disease'
        return 'other'

    def process_edge_df(self, df, src_col='x_index', dst_col='y_index'):
        valid_mask = df[src_col].isin(self.original_to_global) & df[dst_col].isin(self.original_to_global)
        df_valid = df[valid_mask].copy()
        
        if df_valid.empty:
            return {}

        df_valid['src_global'] = df_valid[src_col].map(self.original_to_global)
        df_valid['dst_global'] = df_valid[dst_col].map(self.original_to_global)
        df_valid['src_type'] = df_valid['x_type'].apply(self.sanitize_type)
        df_valid['dst_type'] = df_valid['y_type'].apply(self.sanitize_type)
        
        edges_dict = {}
        
        for (st, dt), group in df_valid.groupby(['src_type', 'dst_type']):
            if st == 'other' or dt == 'other': continue
            
            src_lids = group['src_global'].apply(lambda x: self.global_to_local_type.get(x, (None, None))[1])
            dst_lids = group['dst_global'].apply(lambda x: self.global_to_local_type.get(x, (None, None))[1])
            
            mask = src_lids.notna() & dst_lids.notna()
            src_lids = src_lids[mask].astype(int).values
            dst_lids = dst_lids[mask].astype(int).values
            
            if len(src_lids) > 0:
                edge_t = torch.tensor([src_lids, dst_lids], dtype=torch.long)
                
                # Check relation type
                if 'relation' in group.columns:
                    rel_db = group['relation'].iloc[0]
                else: 
                    rel_db = 'assoc' # Fallback

                # Normalized relation names
                if st == 'drug' and dt == 'gene': rel_name = 'targets'
                elif st == 'gene' and dt == 'drug': rel_name = 'targeted_by'
                elif st == 'gene' and dt == 'disease': rel_name = 'associated_with'
                elif st == 'disease' and dt == 'gene': rel_name = 'associated_with'
                elif st == 'drug' and dt == 'disease': rel_name = 'indication'
                elif st == 'disease' and dt == 'drug': rel_name = 'indicated_for'
                else: rel_name = 'interacts_with'
                
                key = (st, rel_name, dt)
                if key in edges_dict:
                    edges_dict[key] = torch.cat([edges_dict[key], edge_t], dim=1)
                else:
                    edges_dict[key] = edge_t
                    
        return edges_dict

    def build_graph(self, cell_type_file):
        if self.node_mapping is None:
            self.load_node_mapping()
            
        data = HeteroData()
        
        # 1. Setup Nodes
        for t, count in self.type_counts.items():
            data[t].num_nodes = count
            data[t].x = torch.zeros((count, 1)) 
            
        # 2. General Edges
        print("Loading Drug-Gene edges...")
        dg_df = pd.read_csv(os.path.join(self.data_dir, 'edges_drug_gene.csv'))
        dg_edges = self.process_edge_df(dg_df)
        for key, edge_index in dg_edges.items():
            data[key].edge_index = edge_index
            
        print("Loading Gene-Disease edges...")
        gd_df = pd.read_csv(os.path.join(self.data_dir, 'edges_gene_disease.csv'))
        gd_edges = self.process_edge_df(gd_df)
        for key, edge_index in gd_edges.items():
            data[key].edge_index = edge_index

        # 3. Cell-Specific PPI or General PPI
        if cell_type_file == "general":
            # Use PrimeKG's general PPI (Baseline)
            print("Loading General PPI from PrimeKG (Baseline)...")
            ppi_path = os.path.join(self.data_dir, 'edges_ppi_general.csv')
            if os.path.exists(ppi_path):
                ppi_df = pd.read_csv(ppi_path)
                ppi_edges = self.process_edge_df(ppi_df)
                for key, edge_index in ppi_edges.items():
                    # Rename to 'general_ppi' for clarity
                    data['gene', 'general_ppi', 'gene'].edge_index = edge_index
                    # Make symmetric
                    rev = edge_index[[1, 0]]
                    data['gene', 'general_ppi', 'gene'].edge_index = torch.cat([edge_index, rev], dim=1)
            else:
                print(f"Warning: General PPI file {ppi_path} not found.")
        else:
            # Use PINNACLE Cell-Specific PPI
            print(f"Loading Cell-Specific PPI: {cell_type_file}...")
            ppi_path = os.path.join(self.raw_dir, 'pinnacle/networks/ppi_edgelists', cell_type_file)
            if os.path.exists(ppi_path):
                edges_list = []
                with open(ppi_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u_name, v_name = parts[0], parts[1]
                            if u_name in self.gene_symbol_to_global and v_name in self.gene_symbol_to_global:
                                u_global = self.gene_symbol_to_global[u_name]
                                v_global = self.gene_symbol_to_global[v_name]
                                u_local = self.global_to_local_type[u_global][1]
                                v_local = self.global_to_local_type[v_global][1]
                                edges_list.append([u_local, v_local])
                
                if len(edges_list) > 0:
                    edge_index = torch.tensor(edges_list, dtype=torch.long).t()
                    # Symmetric PPI
                    rev_edge_index = edge_index[[1, 0]]
                    full_edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
                    data['gene', 'cell_ppi', 'gene'].edge_index = full_edge_index
            else:
                print(f"Warning: Cell file {ppi_path} not found.")


        # 4. Supervision Labels (Directed Drug->Disease)
        print("Loading Drug-Disease labels...")
        dd_df = pd.read_csv(os.path.join(self.data_dir, 'edges_drug_disease_gold.csv'))
        # LEAKAGE FIX: Directed only
        dd_df = dd_df[(dd_df['x_type'] == 'drug') & (dd_df['y_type'] == 'disease')]
        dd_edges = self.process_edge_df(dd_df)
        for key, edge_index in dd_edges.items():
            data[key].edge_index = edge_index
            
        # 5. Add Reverse Edges for GNN message passing
        self.add_reverse_edges(data)
        
        return data

    def add_reverse_edges(self, data):
        present_types = list(data.edge_types)
        for (src, rel, dst) in present_types:
            if src == dst: 
                pass 
            else:
                rev_key = (dst, f"rev_{rel}", src)
                if rev_key not in data.edge_types:
                    orig_edges = data[src, rel, dst].edge_index
                    rev_edges = orig_edges[[1, 0]]
                    data[rev_key].edge_index = rev_edges

    def save_graph(self, data, name):
        out_path = os.path.join(self.data_dir, f'{name}_graph.pt')
        torch.save(data, out_path)
        print(f"Saved graph to {out_path}")
        return out_path
