import os
import ast
import torch
import pandas as pd
from torch_geometric.data import HeteroData


class GraphBuilder:
    """Build heterogeneous graph from PrimeKG and PINNACLE data."""
    
    def __init__(self, data_dir='data/processed', raw_dir='data/raw'):
        self.data_dir = data_dir
        self.raw_dir = raw_dir
        self.node_mapping = None
        self.global_to_local = {}
        self.original_to_global = {}
        self.gene_name_to_global = {}
        self.gene_name_upper_to_global = {}  # Case-insensitive lookup
        self.type_counts = {}
        self.pinnacle_embeds = None
        self.pinnacle_cell_data = None

    def load_node_mapping(self):
        """Load node ID mappings from processed CSV."""
        path = os.path.join(self.data_dir, 'node_mapping.csv')
        self.node_mapping = pd.read_csv(path)
        
        self.original_to_global = dict(
            zip(self.node_mapping['original_index'], self.node_mapping['node_idx'])
        )
        
        for node_type, group in self.node_mapping.groupby('node_type'):
            clean_type = self._sanitize_type(node_type)
            if clean_type == 'other':
                continue
            sorted_group = group.sort_values('node_idx')
            self.type_counts[clean_type] = len(sorted_group)
            for i, g_idx in enumerate(sorted_group['node_idx']):
                self.global_to_local[g_idx] = (clean_type, i)
        
        # Build gene name mappings (case-sensitive and case-insensitive)
        genes = self.node_mapping[self.node_mapping['node_type'] == 'gene/protein']
        self.gene_name_to_global = dict(zip(genes['node_name'], genes['node_idx']))
        self.gene_name_upper_to_global = dict(
            zip(genes['node_name'].str.upper(), genes['node_idx'])
        )

    def _sanitize_type(self, t):
        """Convert raw node type to standard name."""
        mapping = {'gene/protein': 'gene', 'drug': 'drug', 'disease': 'disease'}
        return mapping.get(t, 'other')

    def load_pinnacle_data(self):
        """Load PINNACLE protein embeddings."""
        if self.pinnacle_embeds is not None:
            return
        
        embed_path = os.path.join(
            self.raw_dir, 'pinnacle/pinnacle_embeds/pinnacle_protein_embed.pth'
        )
        label_path = os.path.join(
            self.raw_dir, 'pinnacle/pinnacle_embeds/pinnacle_labels_dict.txt'
        )
        
        if not os.path.exists(embed_path):
            print(f"PINNACLE embeddings not found: {embed_path}")
            return
        
        self.pinnacle_embeds = torch.load(embed_path, weights_only=False)
        
        with open(label_path, 'r') as f:
            labels = ast.literal_eval(f.read())
        
        cell_types = labels.get('Cell Type', [])
        names = labels.get('Name', [])
        
        self.pinnacle_cell_data = {}
        self.pinnacle_all_proteins = set()
        self.pinnacle_protein_upper = {}  # Case-insensitive protein lookup
        
        idx = 0
        for cell_idx in sorted(self.pinnacle_embeds.keys()):
            n_proteins = self.pinnacle_embeds[cell_idx].shape[0]
            if idx + n_proteins <= len(cell_types):
                cell_name = cell_types[idx]
                proteins = {names[idx + j]: j for j in range(n_proteins)}
                proteins_upper = {names[idx + j].upper(): j for j in range(n_proteins)}
                self.pinnacle_cell_data[cell_name] = {
                    'cell_idx': cell_idx,
                    'proteins': proteins,
                    'proteins_upper': proteins_upper
                }
                self.pinnacle_all_proteins.update(proteins.keys())
            idx += n_proteins
        
        print(f"PINNACLE loaded: {len(self.pinnacle_cell_data)} cell types, "
              f"{len(self.pinnacle_all_proteins)} unique proteins")

    def _get_pinnacle_embed(self, cell_type_file, protein_name):
        """Retrieve PINNACLE embedding for a protein in given cell context."""
        if self.pinnacle_embeds is None or self.pinnacle_cell_data is None:
            return None
        
        cell_name = os.path.splitext(cell_type_file)[0].replace('_', ' ')
        if cell_name not in self.pinnacle_cell_data:
            return None
        
        cell_data = self.pinnacle_cell_data[cell_name]
        
        # Try exact match first, then case-insensitive
        if protein_name in cell_data['proteins']:
            prot_idx = cell_data['proteins'][protein_name]
        elif protein_name.upper() in cell_data['proteins_upper']:
            prot_idx = cell_data['proteins_upper'][protein_name.upper()]
        else:
            return None
        
        cell_idx = cell_data['cell_idx']
        return self.pinnacle_embeds[cell_idx][prot_idx]

    def _lookup_gene(self, name):
        """Look up gene by name with case-insensitive fallback."""
        if name in self.gene_name_to_global:
            return self.gene_name_to_global[name]
        if name.upper() in self.gene_name_upper_to_global:
            return self.gene_name_upper_to_global[name.upper()]
        return None

    def _process_edges(self, df):
        """Convert edge DataFrame to typed edge tensors."""
        valid = (
            df['x_index'].isin(self.original_to_global) & 
            df['y_index'].isin(self.original_to_global)
        )
        df = df[valid].copy()
        if df.empty:
            return {}
        
        df['src_global'] = df['x_index'].map(self.original_to_global)
        df['dst_global'] = df['y_index'].map(self.original_to_global)
        df['src_type'] = df['x_type'].apply(self._sanitize_type)
        df['dst_type'] = df['y_type'].apply(self._sanitize_type)
        
        edges = {}
        for (st, dt), group in df.groupby(['src_type', 'dst_type']):
            if st == 'other' or dt == 'other':
                continue
            
            src = group['src_global'].apply(
                lambda x: self.global_to_local.get(x, (None, None))[1]
            )
            dst = group['dst_global'].apply(
                lambda x: self.global_to_local.get(x, (None, None))[1]
            )
            
            mask = src.notna() & dst.notna()
            src = src[mask].astype(int).values
            dst = dst[mask].astype(int).values
            
            if len(src) == 0:
                continue
            
            rel = self._get_relation_name(st, dt)
            key = (st, rel, dt)
            edge_tensor = torch.tensor([src, dst], dtype=torch.long)
            
            if key in edges:
                edges[key] = torch.cat([edges[key], edge_tensor], dim=1)
            else:
                edges[key] = edge_tensor
        
        return edges

    def _get_relation_name(self, src_type, dst_type):
        """Get canonical relation name for edge type."""
        rel_map = {
            ('drug', 'gene'): 'targets',
            ('gene', 'drug'): 'targeted_by',
            ('gene', 'disease'): 'associates',
            ('disease', 'gene'): 'associates',
            ('drug', 'disease'): 'indication',
            ('disease', 'drug'): 'indicated_for',
        }
        return rel_map.get((src_type, dst_type), 'interacts')

    def build(self, cell_type_file, use_pinnacle=True):
        """Build heterogeneous graph for given cell type context."""
        if self.node_mapping is None:
            self.load_node_mapping()
        
        if use_pinnacle and cell_type_file != 'general':
            self.load_pinnacle_data()
        
        data = HeteroData()
        
        # Initialize nodes
        for t, count in self.type_counts.items():
            data[t].num_nodes = count
        
        # Load edges
        dg_df = pd.read_csv(os.path.join(self.data_dir, 'edges_drug_gene.csv'))
        for key, idx in self._process_edges(dg_df).items():
            data[key].edge_index = idx
        
        gd_df = pd.read_csv(os.path.join(self.data_dir, 'edges_gene_disease.csv'))
        for key, idx in self._process_edges(gd_df).items():
            data[key].edge_index = idx
        
        # Load PPI with case-insensitive matching
        if cell_type_file == 'general':
            ppi_path = os.path.join(self.data_dir, 'edges_ppi_general.csv')
            if os.path.exists(ppi_path):
                ppi_df = pd.read_csv(ppi_path)
                for key, idx in self._process_edges(ppi_df).items():
                    rev = idx[[1, 0]]
                    data['gene', 'ppi', 'gene'].edge_index = torch.cat([idx, rev], dim=1)
        else:
            ppi_path = os.path.join(
                self.raw_dir, 'pinnacle/networks/ppi_edgelists', cell_type_file
            )
            if os.path.exists(ppi_path):
                edges = []
                with open(ppi_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            u, v = parts[0], parts[1]
                            u_g = self._lookup_gene(u)
                            v_g = self._lookup_gene(v)
                            if u_g is not None and v_g is not None:
                                u_l = self.global_to_local[u_g][1]
                                v_l = self.global_to_local[v_g][1]
                                edges.append([u_l, v_l])
                
                if edges:
                    idx = torch.tensor(edges, dtype=torch.long).t()
                    data['gene', 'ppi', 'gene'].edge_index = torch.cat(
                        [idx, idx[[1, 0]]], dim=1
                    )

        # Load labels
        dd_df = pd.read_csv(os.path.join(self.data_dir, 'edges_drug_disease_gold.csv'))
        dd_df = dd_df[(dd_df['x_type'] == 'drug') & (dd_df['y_type'] == 'disease')]
        for key, idx in self._process_edges(dd_df).items():
            data[key].edge_index = idx
        
        # Add reverse edges
        for src, rel, dst in list(data.edge_types):
            if src != dst:
                rev_key = (dst, f'rev_{rel}', src)
                if rev_key not in data.edge_types:
                    data[rev_key].edge_index = data[src, rel, dst].edge_index[[1, 0]]
        
        # Build node features with case-insensitive ID matching
        embed_dim = 128
        for node_type in data.node_types:
            n = data[node_type].num_nodes
            if node_type == 'gene' and use_pinnacle and cell_type_file != 'general':
                feats = torch.zeros(n, embed_dim)
                genes = self.node_mapping[
                    self.node_mapping['node_type'] == 'gene/protein'
                ].sort_values('node_idx')['node_name'].tolist()
                
                matched = 0
                for i, name in enumerate(genes[:n]):
                    emb = self._get_pinnacle_embed(cell_type_file, name)
                    if emb is not None:
                        feats[i] = emb
                        matched += 1
                
                print(f"[ID Alignment] Gene features: {matched}/{n} matched "
                      f"({100*matched/n:.1f}%)")
                
                if matched < n * 0.01:
                    print("[Warning] Very low match rate! Check ID mapping.")
                
                data[node_type].x = feats
            else:
                data[node_type].x = torch.zeros(n, embed_dim)
        
        return data

    def save(self, data, name):
        """Save graph to disk."""
        path = os.path.join(self.data_dir, f'{name}_graph.pt')
        torch.save(data, path)
        return path
