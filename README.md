# Cell-Type Specific Drug Repurposing

A GNN-based drug repurposing framework that integrates cell-type specific PPI networks from [PINNACLE](https://www.nature.com/articles/s41592-024-02469-9) with biomedical knowledge from [PrimeKG](https://www.nature.com/articles/s41597-022-01809-5). This project implements key methodologies from [TxGNN](https://www.nature.com/articles/s41591-024-03122-z) for zero-shot drug-disease prediction.

## Features

### Graph Construction
- **Heterogeneous graph** with drug, disease, and gene nodes from PrimeKG
- **Multi-relational PPI fusion**: Combines generic PPI edges with cell-specific edges to avoid isolated nodes while preserving cellular context
- **PINNACLE embeddings**: 128-dimensional protein representations for 132 cell types

### Model Architecture
- **HeteroGCN encoder**: Heterogeneous graph neural network with SAGEConv layers
- **Lazy feature projection**: Per-node-type linear layers that adapt to different input dimensions (128-dim PINNACLE vs 384-dim text embeddings)
- **SimGNN decoder**: Similarity-based link predictor for zero-shot inference (TxGNN-style)

### Evaluation Metrics
- AUROC, AUPRC, MRR, Recall@K
- **NS-Recall**: Normalized Sensitivity Recall from TxGNN to address popularity bias

## Installation

```bash
pip install -r requirements.txt
```

**Requirements**: PyTorch, PyTorch Geometric, sentence-transformers, pandas, numpy, scikit-learn

## Quick Start

```bash
# 1. Preprocess data and build graph
python scripts/run_preprocess.py --config configs/config_base.yaml

# 2. Train model
python scripts/run_train.py --config configs/config_base.yaml

# 3. Evaluate
python scripts/run_eval.py --config configs/config_base.yaml
```

## Project Structure

```
├── configs/                 # Experiment configurations
├── data/
│   ├── raw/                # PrimeKG and PINNACLE data
│   │   └── pinnacle/       # Cell-type specific networks & embeddings
│   ├── processed/          # Built graph objects
│   └── embeddings/         # Cached text embeddings
├── notebooks/              # Analysis notebooks
├── scripts/                # Entry point scripts
├── src/
│   ├── data/              # GraphBuilder, DatasetLoader, TextEmbedder
│   ├── models/            # HeteroGCN, LinkPredictor (with SimGNN)
│   ├── training/          # Trainer class
│   ├── evaluation/        # AUROC, AUPRC, MRR, NS-Recall
│   └── utils/             # Helpers and logging
└── results/               # Checkpoints and logs
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `01_data_exploration.ipynb` | PrimeKG and PINNACLE data statistics |
| `02_graph_construction.ipynb` | Graph building, multi-relational PPI, split verification |
| `03_result_visualization.ipynb` | ROC/PR curves, score distributions |
| `04_case_study_hepatocyte.ipynb` | Hepatocyte vs General model comparison |

## Configurations

| Config | Cell Type | Split | Features |
|--------|-----------|-------|----------|
| `config_base.yaml` | Microglial | Random | PINNACLE embeddings |
| `config_general.yaml` | General | Random | Baseline (no cell context) |
| `config_liver.yaml` | Hepatocyte | Random | PINNACLE embeddings |
| `config_zeroshot.yaml` | Microglial | Disease | ID-based embeddings |
| `config_zeroshot_textembed.yaml` | Microglial | Disease | Text embeddings (384-dim) |
| `config_zeroshot_simgnn.yaml` | Microglial | Disease | Text + SimGNN decoder |

## Split Strategies

- **Random**: Standard edge-level split for transductive learning (train/val/test = 0.8/0.1/0.1)
- **Disease**: Hold out entire diseases for zero-shot evaluation (min 2 edges per disease)

## Key Results

### Transductive Learning (Random Split)

| Configuration | AUROC | AUPRC | MRR | R@50 |
|--------------|-------|-------|-----|------|
| Microglial cell-specific | 0.92 | 0.92 | 0.17 | 0.98 |
| General baseline | **0.96** | **0.96** | 0.22 | 1.00 |
| Hepatocyte cell-specific | 0.93 | 0.93 | **0.21** | 1.00 |

> **Key finding**: Hepatocyte model achieves comparable AUROC to baseline while improving MRR for liver-related diseases.

### Zero-Shot Learning (Disease Split)

| Configuration | AUROC | AUPRC | Improvement |
|--------------|-------|-------|-------------|
| ID embeddings only | 0.67 | 0.62 | baseline |
| + Text embeddings | 0.72 | 0.65 | +7.5% AUROC |
| + SimGNN decoder | **0.78** | **0.68** | **+16.4% AUROC** |

> **Key finding**: Text embeddings enable true zero-shot prediction for unseen diseases. SimGNN decoder further improves by leveraging disease similarity.

## References

- Huang et al. **TxGNN enables zero-shot prediction of therapeutic use of drug candidates.** *Nature Medicine* (2024) [[paper]](https://www.nature.com/articles/s41591-024-03122-z)
- Li et al. **PINNACLE: Context-aware gene representations.** *Nature Methods* (2024) [[paper]](https://www.nature.com/articles/s41592-024-02469-9)
- Chandak et al. **Building a knowledge graph to enable precision medicine.** *Scientific Data* (2022) [[paper]](https://www.nature.com/articles/s41597-022-01809-5)

## License

MIT License
