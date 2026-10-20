# CNN for multilabel classification

Kaggle [cats-and-dogs](https://www.kaggle.com/competitions/cats-and-dogs). Each photo has 5 labels: Cat, Dog, Moris, Motya, Biatrix.
A custom CNN in PyTorch Lightning, runs tracked in MLflow.

## Structure

```
data/            Kaggle data (not in git) + folds.csv
docs/            experiment notes, metric choice
notebooks/       EDA, custom_cnn
src/
├── config.py      paths, label order
├── split.py       5 stratified folds → data/folds.csv
├── data.py        augmentations, Dataset, DataModule
├── classifier.py  LightningModule: loss, F1, optimizer, scheduler
├── cnn.py         network + train()
├── postprocess.py TTA, pHash blend, thresholds, hierarchy
└── tracking.py    MLflow setup
models/          checkpoints, one folder per run
mlruns/          MLflow database and artifacts
```

## Setup

```bash
uv sync
kaggle competitions download -c cats-and-dogs -p data
unzip data/cats-and-dogs.zip -d data && rm data/cats-and-dogs.zip
```

`data/folds.csv` is in git. Rebuild it only if the split changes: `uv run python -m src.split`.

## Train

```bash
uv run python -m src.cnn
```

- Trains on folds 1–4 and validates on fold 0.
- Normalisation mean/std are computed on the training part.
- The best checkpoint by val macro F1 goes to `models/<run_id>/`.

## Experiments

`notebooks/custom_cnn.ipynb` is the experiment log: each section trains (or reuses a checkpoint), writes `models/<run_id>/submission*.csv` and submits it with the kaggle CLI.

1. Baseline CNN, threshold 0.5.
2. Post-processing on the same checkpoint: TTA hflip → pHash ≤ 2 blend with train labels → per-class thresholds on fold 0 → named animals imply their species.

## MLflow

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

Experiment `dl_lab-2`. Each run has hyperparameters, losses, macro and per-class F1, LR, gradient norms per layer, architecture and source code.
