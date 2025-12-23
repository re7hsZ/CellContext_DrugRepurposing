# CellContext_DrugRepurposing

**Context-Aware Drug Repurposing using Graph Neural Networks**

This project implements a GNN-based drug repurposing model that integrates cell-type specific PPI networks from [PINNACLE](https://www.nature.com/articles/s41592-024-02341-3) with the [PrimeKG](https://www.nature.com/articles/s41591-024-03233-x) biomedical knowledge graph.

## Project Overview

Traditional drug repurposing models use **general PPI networks** that ignore the context of different cell types (e.g., hepatocytes vs. neurons). This project addresses this limitation by:

1. Using **PINNACLE** to generate cell-type specific protein-protein interaction networks.
2. Combining these with **PrimeKG**'s Drug-Gene and Gene-Disease relationships.
3. Training a **Heterogeneous Graph Convolutional Network (HeteroGCN)** on these context-aware graphs.
4. Predicting **Drug-Indication** relationships.

## Project Structure

```
CellContext_DrugRepurposing/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── configs/                    # Experiment configurations (YAML)
│   ├── config_base.yaml        # Default configuration
│   ├── config_liver.yaml       # Hepatocyte-specific experiment
│   ├── config_neuron.yaml      # Neuron-specific experiment
│   └── config_general.yaml     # Baseline (General PPI)
│
├── data/
│   ├── raw/                    # Original data (read-only)
│   │   ├── primekg/            # PrimeKG CSV files (kg.csv)
│   │   └── pinnacle/           # PINNACLE cell-specific PPIs
│   ├── processed/              # Built graph objects (.pt)
│   └── mappings/               # ID mapping tables
│
├── notebooks/                  # Jupyter notebooks for analysis
│
├── scripts/                    # Entry point scripts
│   ├── run_preprocess.py       # Build graph from raw data
│   ├── run_train.py            # Train model
│   └── run_eval.py             # Evaluate saved model
│
├── src/                        # Source code
│   ├── data/                   # Data processing pipeline
│   │   ├── preprocess_primekg.py   # Parse raw kg.csv
│   │   ├── graph_builder.py        # Fuse PrimeKG + PINNACLE -> PyG graph
│   │   ├── dataset_loader.py       # Load .pt graph files
│   │   ├── splitter.py             # Train/Val/Test split
│   │   └── id_mapper.py            # ID alignment utilities
│   │
│   ├── models/                 # Model definitions
│   │   ├── gcn_encoder.py          # HeteroGCN encoder
│   │   └── link_predictor.py       # Dot-product link predictor
│   │
│   ├── training/               # Training logic
│   │   ├── trainer.py              # Training loop
│   │   ├── loss.py                 # Loss functions (BCE)
│   │   └── negative_sampler.py     # Negative sampling
│   │
│   ├── evaluation/             # Metrics
│   │   └── metrics.py              # AUROC, AUPRC, MRR, Recall@K
│   │
│   └── utils/                  # Utilities
│       ├── helpers.py              # Config loading, seed setting
│       └── logger.py               # Logging setup
│
└── results/                    # Output (auto-generated)
    ├── logs/                   # Training logs
    ├── checkpoints/            # Model weights (.pth)
    └── predictions/            # Prediction outputs
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data:**
   - Place PrimeKG's `kg.csv` in `data/raw/primekg/`.
   - Place PINNACLE PPI files in `data/raw/pinnacle/networks/ppi_edgelists/`.
   - Run the PrimeKG preprocessing:
     ```bash
     python src/data/preprocess_primekg.py
     ```

3. **Build Graph (Preprocessing):**
   ```bash
   python scripts/run_preprocess.py --config configs/config_base.yaml
   ```

4. **Train Model:**
   ```bash
   python scripts/run_train.py --config configs/config_base.yaml
   ```

5. **Evaluate Model:**
   ```bash
   python scripts/run_eval.py --config configs/config_base.yaml
   ```

## Running Different Experiments

### Cell-Type Specific Experiment
```bash
# Preprocess for liver (hepatocyte) context
python scripts/run_preprocess.py --config configs/config_liver.yaml
python scripts/run_train.py --config configs/config_liver.yaml
```

### Baseline (General PPI)
```bash
# Use PrimeKG's general PPI instead of cell-specific
python scripts/run_preprocess.py --config configs/config_general.yaml
python scripts/run_train.py --config configs/config_general.yaml
```

## Evaluation Metrics

- **auROC**: Area under ROC curve (classification performance)
- **auPRC**: Area under Precision-Recall curve (important for class imbalance)
- **F1 Score**: Harmonic mean of precision and recall
- **MRR**: Mean Reciprocal Rank (ranking quality)
- **Recall@K**: Fraction of positives ranked in top K

## References

- **TxGNN**: Huang et al., *A foundation model for clinician-centered drug repurposing*, Nature Medicine (2024).
- **PINNACLE**: Li et al., *Contextual AI models for single-cell protein biology*, Nature Methods (2024).
- **PrimeKG**: Chandak et al., *Precision medicine knowledge graph*, Scientific Data (2023).
