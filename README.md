# ML Systems Labs

Individual lab work for the ML Systems course (textbook: Reddi, *Introduction to Machine Learning Systems*).

## Structure
```
ml-labs/
  README.md
  requirements.txt
  .gitignore
  Lab1/  report.md  src/  results/
  Lab2/  ...
  ...
  Lab11/ ...
  Project/ ...
```

## Setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Lab 1
```bash
cd Lab1
python src/print_versions.py > results/versions.txt
python src/measure.py
```
See `Lab1/report.md`.

## Lab 2
```bash
cd Lab2
python src/lab2.py
```
See `Lab2/report.md`.

## Lab 3
```bash
cd Lab3
python src/lab3.py
```
See `Lab3/report.md`.

## Lab 4
```bash
cd Lab4
python src/run_pipeline.py
```
See `Lab4/report.md` and `Lab4/data_card.md`.

## Lab 5
```bash
cd Lab5
python src/lab5.py        # about 5 minutes: the scalar engine is slow on purpose (it is timed 6 times)
```
Scalar autograd engine (`src/engine.py`), gradient checks, make_moons training vs PyTorch, eager vs traced vs compiled, non-contiguous tensors. Results in `Lab5/results/`, write-up in `Lab5/report.md`.

## Lab 6
```bash
cd Lab6
python src/train.py --config config.yaml --name base       # one pipeline run (YAML-driven, MLflow, checkpoints)
python src/studies.py all                                    # profile, workers, batch size, schedules, accumulation, resume (about 30 min)
mlflow ui --backend-store-uri mlruns                         # browse the runs
```
Pipeline, studies and results in `Lab6/`, write-up in `Lab6/report.md`, project proposal in `Lab6/proposal.md`.

## Lab 7
```bash
cd Lab7
python src/lab7.py        # baseline, pruning, latency/size, distillation, summary table (about 20-30 min)
```
Baseline CNN (1.03 M parameters), global / layer-wise / structured pruning, sparse-format size and kernel test, distillation into a 106 k-parameter student. Results in `Lab7/results/`, write-up in `Lab7/report.md`.

## Lab 8
```bash
cd Lab8
python src/lab8.py        # manual quantization, dynamic + static quantization, ONNX export, variant table (about 10 min)
```
Results in `Lab8/results/`, write-up in `Lab8/report.md`.

## Project
Baseline milestone of the individual project: data pipeline (from Lab 4), trained baseline model and `metrics.json`; see `Project/README.md`.
