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
