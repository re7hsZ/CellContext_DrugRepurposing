# Cell-Type Specific Drug Repurposing

A GNN-based drug repurposing framework integrating cell-type specific PPI networks.

## Overview

This project combines general biomedical knowledge from PrimeKG with cell-type specific protein interactions from PINNACLE to predict drug-disease indications using graph neural networks.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Preprocess data
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
│   ├── raw/                # Original PrimeKG and PINNACLE data
│   └── processed/          # Built graph objects
├── scripts/                # Entry point scripts
├── src/
│   ├── data/              # Data loading and processing
│   ├── models/            # GNN model definitions
│   ├── training/          # Training utilities
│   ├── evaluation/        # Metrics
│   └── utils/             # Helpers
└── results/               # Outputs
```

## Configurations

| Config | Description |
|--------|-------------|
| `config_base.yaml` | Default with cell-specific PPI |
| `config_general.yaml` | Baseline with general PPI |
| `config_liver.yaml` | Hepatocyte context |
| `config_zeroshot.yaml` | Zero-shot disease evaluation |

## References

- TxGNN: Nature Medicine (2024)
- PINNACLE: Nature Methods (2024)
