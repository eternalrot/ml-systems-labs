# Project: Adult income classifier (baseline milestone)

![python](https://img.shields.io/badge/python-3.11-blue) ![pytorch](https://img.shields.io/badge/PyTorch-2.3.1-orange) ![status](https://img.shields.io/badge/status-baseline-green)

Baseline model and data pipeline for the individual project of the ML Systems course (Vistula University, Pavlo Budko, ID 72591).

## Layout
```
Project/
  data_pipeline/   Lab 4 pipeline (ingest, validation, leakage-free cleaning, hash, data card)
  src/train_baseline.py
  models/baseline.pt        trained baseline (MLP 107-128-64-1)
  metrics.json              accuracy, F1, AUC, size, latency, data hashes, environment
```

## Run
```bash
cd Project
python data_pipeline/src/run_pipeline.py     # writes data/processed/*.csv (not committed)
python src/train_baseline.py                 # writes models/baseline.pt and metrics.json
```

## Constraints (from the proposal)
Latency at most 50 ms and model size at most 5 MB; `metrics.json` records both and whether they are met.
