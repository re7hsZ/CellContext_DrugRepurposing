# Cell-Type Specific Drug Repurposing

GNN-based drug repurposing framework integrating cell-type specific PPI networks from PINNACLE with biomedical knowledge from PrimeKG.

## Features

- Heterogeneous graph neural network for drug-disease link prediction
- Cell-type specific protein-protein interaction networks
- PINNACLE protein embeddings integration
- Multi-relational PPI fusion (generic + cell-specific edges)
- Text embeddings for zero-shot disease prediction
- SimGNN-style similarity decoder for inductive inference
- Comprehensive evaluation metrics (AUROC, AUPRC, MRR, Recall@K, NS-Recall)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Preprocess Data

```bash
python scripts/run_preprocess.py --config configs/config_base.yaml
```

### 2. Train Model

```bash
python scripts/run_train.py --config configs/config_base.yaml
```

### 3. Evaluate

```bash
python scripts/run_eval.py --config configs/config_base.yaml
```

## Project Structure

```
├── configs/                 # Experiment configurations
├── data/
│   ├── raw/                # PrimeKG and PINNACLE data
│   ├── processed/          # Built graph objects
│   └── embeddings/         # Cached text embeddings
├── notebooks/              # Analysis notebooks
├── scripts/                # Entry point scripts
├── src/
│   ├── data/              # Data loading and graph building
│   ├── models/            # GNN encoder and link predictor
│   ├── training/          # Training loop
│   ├── evaluation/        # Metrics computation
│   └── utils/             # Helpers and logging
└── results/               # Checkpoints and logs
```

## Configurations

| Config | Description |
|--------|-------------|
| `config_base.yaml` | Cell-specific PPI with PINNACLE features |
| `config_general.yaml` | Baseline with general PPI |
| `config_liver.yaml` | Hepatocyte cell context |
| `config_zeroshot.yaml` | Zero-shot disease evaluation |
| `config_zeroshot_textembed.yaml` | Zero-shot with text embeddings |
| `config_zeroshot_simgnn.yaml` | Zero-shot with SimGNN decoder |

## Split Strategies

- **Random**: Standard edge-level split for transductive learning
- **Disease**: Hold out entire diseases for inductive/zero-shot evaluation

## Key Results

| Configuration | AUROC | AUPRC | MRR |
|--------------|-------|-------|-----|
| Cell-specific (microglial) | 0.93 | 0.91 | 0.16 |
| Cell-specific (hepatocyte) | 0.94 | 0.94 | 0.27 |
| General baseline | 0.96 | 0.96 | 0.22 |
| Zero-shot (ID only) | 0.48 | 0.52 | 0.02 |
| Zero-shot (text embed) | 0.75 | 0.67 | 0.01 |

## References

- Huang et al. TxGNN enables zero-shot prediction of therapeutic use. Nature Medicine (2024)
- Li et al. PINNACLE: Context-aware gene representations. Nature Methods (2024)


