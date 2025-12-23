import pandas as pd
import os
import numpy as np

# ================= 配置路径 =================
RAW_PATH = 'data/raw/primekg/kg.csv'
OUTPUT_DIR = 'data/processed'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_primekg():
    print(f"Loading PrimeKG from {RAW_PATH}...")
    # 读取所有数据 (low_memory=False 防止混合类型警告)
    df = pd.read_csv(RAW_PATH, low_memory=False)
    
    # 打印一下所有的关系类型，确保我们没漏掉什么
    print("All relation types found:", df['relation'].unique())
    print("All node types found:", df['x_type'].unique())

    # ================= 1. 提取节点信息 (建立 ID 索引) =================
    # 我们需要构建一个全局的节点映射表: PrimeKG_ID -> Index (0, 1, 2...)
    # 这样后续放入 PyTorch Geometric 时才能用矩阵计算
    print("Extracting node information...")
    
    # 获取 x 和 y 的所有节点信息
    nodes_x = df[['x_index', 'x_id', 'x_type', 'x_name', 'x_source']].rename(
        columns={'x_index': 'original_index', 'x_id': 'node_id', 'x_type': 'node_type', 'x_name': 'node_name', 'x_source': 'source'})
    nodes_y = df[['y_index', 'y_id', 'y_type', 'y_name', 'y_source']].rename(
        columns={'y_index': 'original_index', 'y_id': 'node_id', 'y_type': 'node_type', 'y_name': 'node_name', 'y_source': 'source'})
    
    # 合并去重
    all_nodes = pd.concat([nodes_x, nodes_y]).drop_duplicates(subset=['node_id', 'node_type']).reset_index(drop=True)
    
    # 为每个节点分配一个新的连续索引 (0, 1, 2, ..., N)
    all_nodes['node_idx'] = all_nodes.index
    
    # 保存节点表，这在后续 ID Mapping 时非常重要
    all_nodes.to_csv(os.path.join(OUTPUT_DIR, 'node_mapping.csv'), index=False)
    print(f"Saved {len(all_nodes)} unique nodes to node_mapping.csv")

    # ================= 2. 拆分边 (核心步骤) =================
    print("Splitting edges...")
    
    # A. 提取 药物-靶点 (Drug-Protein) 边
    # 这里的关系通常是 drug_protein
    edge_drug_gene = df[
        ((df['x_type'] == 'drug') & (df['y_type'] == 'gene/protein')) | 
        ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'drug'))
    ].copy()
    edge_drug_gene.to_csv(os.path.join(OUTPUT_DIR, 'edges_drug_gene.csv'), index=False)
    print(f"Extracted {len(edge_drug_gene)} Drug-Gene edges.")

    # B. 提取 基因-疾病 (Gene-Disease) 边
    # 这里的关系通常是 contraindication (这是药的) 或 disgenet 里的 gene_disease
    # PrimeKG 里基因和疾病的关系主要是 'gene_disease' 或者是从 GDA (Gene Disease Association) 来的
    # 我们通过节点类型筛选最保险
    edge_gene_disease = df[
        ((df['x_type'] == 'gene/protein') & (df['y_type'] == 'disease')) | 
        ((df['x_type'] == 'disease') & (df['y_type'] == 'gene/protein'))
    ].copy()
    edge_gene_disease.to_csv(os.path.join(OUTPUT_DIR, 'edges_gene_disease.csv'), index=False)
    print(f"Extracted {len(edge_gene_disease)} Gene-Disease edges.")

    # C. 提取 标签 (Label): 药物-疾病 (Drug-Disease) 边
    # 这些是我们要预测的目标 (Indication)
    # 注意：PrimeKG 里可能包含 contraindication (禁忌症)，我们要只要 'indication'
    # 你需要根据打印出来的 relation 列表确认一下是否包含 'contraindication'
    edge_drug_disease = df[
        ((df['x_type'] == 'drug') & (df['y_type'] == 'disease')) | 
        ((df['x_type'] == 'disease') & (df['y_type'] == 'drug'))
    ].copy()
    
    # 简单过滤，只保留 indication (适应症)
    if 'indication' in df['relation'].unique():
        edge_drug_disease = edge_drug_disease[edge_drug_disease['relation'] == 'indication']
        
    edge_drug_disease.to_csv(os.path.join(OUTPUT_DIR, 'edges_drug_disease_gold.csv'), index=False)
    print(f"Extracted {len(edge_drug_disease)} Drug-Disease indication edges (Ground Truth).")

    # D. 通用 PPI (我们后面要丢弃或替换的部分)
    edge_ppi = df[df['relation'] == 'protein_protein'].copy()
    edge_ppi.to_csv(os.path.join(OUTPUT_DIR, 'edges_ppi_general.csv'), index=False)
    print(f"Extracted {len(edge_ppi)} General PPI edges (to be replaced by PINNACLE).")

if __name__ == "__main__":
    process_primekg()