# Reproducible Tabular Classification with PyTorch

A small, reproducible PyTorch project for binary classification on the UCI Adult Income dataset.

## Project goal

The goal is to build a compact end-to-end workflow that demonstrates practical ML engineering: mixed-type tabular preprocessing, PyTorch training, reproducible experiment configuration, checkpointing, metric logging, evaluation, experiment comparison, and inference.

This is not a project about the Adult Income dataset itself. The dataset provides a manageable problem on which to build trust through a clean, reproducible workflow. The engineering process and the ability to rerun and inspect experiments are the main deliverables.

## Why this project exists

Many small ML examples stop at a notebook and a single model result. This project instead focuses on the surrounding workflow required to make experiments understandable and repeatable while deliberately avoiding production-platform complexity.

## MVP scope

The planned MVP includes:

- numerical and categorical feature preprocessing fitted on training data only;
- a reproducible train/validation split and the official Adult test split;
- a PyTorch linear baseline and two small MLP configurations;
- YAML-based experiment configuration;
- best and latest model checkpoints;
- per-epoch and final metric logging;
- separate evaluation and single-record prediction commands;
- comparison of saved experiment results;
- focused tests for the most important data and model contracts.

## Planned CLI flow

The command interfaces are placeholders during Step 1. The intended workflow is:

```bash
python train.py --config configs/linear.yaml
python train.py --config configs/mlp.yaml
python train.py --config configs/mlp_dropout.yaml

python evaluate.py --run-dir artifacts/runs/<run-id> --split test
python compare.py --runs-dir artifacts/runs
python predict.py --run-dir artifacts/runs/<run-id> --input examples/person.json
```

## Data contract

- `adult.data` is the original training source and `adult.test` is the official final test set from the UCI Adult dataset.
- The validation split is created only from `adult.data`, with target stratification and the configured random seed.
- The official test set is reserved for final evaluation and must not be used for model selection.
- A `?` value is treated as missing and surrounding whitespace is removed from string columns.
- The target maps `<=50K` and `<=50K.` to `0`, and `>50K` and `>50K.` to `1`.
- Model preprocessing will be fitted only on the training split in the next project step.

## Intentionally out of scope

- Docker and Kubernetes
- cloud deployment and managed training
- MLflow, Weights & Biases, Hydra, and DVC
- FastAPI, a frontend, or another serving layer
- a notebook-driven primary workflow
- distributed training or a general-purpose training framework
- large hyperparameter searches and automated tuning

## Current status

**Step 2 - data contract and loader.**

The official UCI Adult files can be downloaded and loaded, raw values are cleaned according to the documented contract, and a deterministic stratified validation split is created from the original training source. Model preprocessing, training, evaluation, comparison, and prediction are not implemented yet.

