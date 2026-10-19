# CNN for multilabel classification

## Setup

```bash
uv sync
```

## Data

Download the [cats-and-dogs](https://www.kaggle.com/competitions/cats-and-dogs/data) competition data and unpack it into `data/`:

```bash
kaggle competitions download -c cats-and-dogs -p data
unzip data/cats-and-dogs.zip -d data && rm data/cats-and-dogs.zip
```

Expected layout:

```
data/
├── train/
├── test/
├── train.csv
├── test.csv
└── sample_submission.csv
```

## MLflow

Runs are stored in `mlruns/` at the repo root. To open the UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```
