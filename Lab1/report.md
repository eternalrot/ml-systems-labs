# Lab 1: Environment and first system measurements

## 1. Goal
Set up a reproducible, pinned Python environment, train two baseline models (linear and tree ensemble) on Breast Cancer Wisconsin, measure their *system* cost (training time, latency, size, memory), and decide on which deployment targets (Cloud / Edge / Mobile / TinyML) each model fits.

## 2. Method
- Environment: Python 3.11 virtualenv, packages pinned in `requirements.txt`; `src/print_versions.py` writes all versions to `results/versions.txt`.
- Data: `load_breast_cancer(return_X_y=True)`, 569 samples, 30 features; 70/30 stratified split, `random_state=42`. All seeds set to 42.
- Models: `LogisticRegression(max_iter=1000, random_state=42)` and `RandomForestClassifier(n_estimators=100, random_state=42)`.
- Training time: 1 warm-up fit, then median of 5 fits.
- Single-sample inference: 1 warm-up call, then median of 100 `predict` calls on one test row.
- Model size: `joblib.dump` to `results/`, size from file system (bytes and KB).
- Peak memory: process RSS (`psutil.Process().memory_info().rss`) sampled every 1 ms during training and during 100 inferences. Both absolute peak RSS and the increase over the pre-measurement baseline are recorded.
- Everything is produced by `src/measure.py`; raw numbers are in `results/system_cost.csv`.

## 3. Results

| Model | Test accuracy | Train time (median of 5) | Latency / sample (median of 100) | Model size | Peak RSS train | Peak RSS infer | Extra RSS train / infer |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.9474 | 250.9 ms | 0.033 ms | 1055 B (1.03 KB) | 110.5 MB | 110.5 MB | 0.09 / 0.02 MB |
| Random Forest | 0.9357 | 102.1 ms | 1.407 ms | 290 889 B (284.1 KB) | 112.6 MB | 112.6 MB | 0.32 / 0.00 MB |

Deployment fit (budgets from the task; full table in `results/budget_fit.csv`):

| Target | Budget (memory / latency / size) | Logistic Regression | Random Forest |
|---|---|---|---|
| Cloud | ≥1 GB / ≤100 ms / ≤500 MB | Fits | Fits |
| Edge | 256–1024 MB / ≤50 ms / ≤50 MB | Fits | Fits |
| Mobile | 64–256 MB / ≤20 ms / ≤10 MB | Fits (110 MB process) | Fits (113 MB process) |
| TinyML | ≤256 KB / ≤10 ms / ≤100 KB | Latency and size OK (1 KB), **memory fails** with a Python runtime | **Fails**: 284 KB model > 100 KB, and memory |

## 4. Deployment-fit argument
Both models run comfortably on Cloud and Edge: latency is 2–3 orders of magnitude below the limits and model size is far below 50 MB. On Mobile the models also fit: the ~110–113 MB figure is the whole Python process (interpreter, NumPy, scikit-learn), while the models themselves add well under 1 MB, so latency (≤1.4 ms vs 20 ms) and size (≤284 KB vs 10 MB) leave a large margin. TinyML is the interesting case. Logistic regression is only 1 KB and has microsecond latency, and its inference is 30 multiply-adds, so it could run on a microcontroller if the weights are exported to C; but the scikit-learn/Python process itself needs ~110 MB, so the measured setup does not meet the 256 KB memory limit. The random forest cannot go to TinyML at all: 284 KB exceeds the 100 KB size limit even before runtime memory is counted.

## 5. Conclusions
1. The logistic regression turned out to be the better baseline for this dataset. It has slightly higher accuracy (0.9474 vs 0.9357), predicts one sample about 40 times faster (0.033 ms vs 1.4 ms) and its file is about 275 times smaller (1 KB vs 284 KB). The forest is only faster to train (102 ms vs 251 ms), and that happens because LogisticRegression runs all 1000 iterations on unscaled data. Training happens once, while latency and size matter every time the model is deployed, so the forest's extra cost is not justified here.
2. The memory I measured (about 110-113 MB) is almost entirely Python, NumPy and scikit-learn, not the model: the models themselves add only about 0.1-0.3 MB. So the "memory" number depends on how you deploy. A Python service needs about 110 MB, but the model alone (weights) needs a few KB. This is why the same model can fit Mobile with the Python runtime and still be a candidate for TinyML after export.
3. Hardware limits, not accuracy, decide where a model can run. Both models fit Cloud, Edge and Mobile with a large margin. For TinyML the random forest fails immediately because 284 KB is more than the 100 KB limit, while the logistic regression (1 KB) could fit only if it is exported without Python (for example as C code with 30 weights). Also, the accuracy gap is small and the test set only has 171 samples, so I would choose the model by cost, not by 1 percentage point of accuracy.

## Notes and limitations
- Timings are machine-dependent and the test set is small (171 samples, so one sample ≈ 0.6 pp accuracy); the accuracy difference between the models is within noise.
- `LogisticRegression` emits a ConvergenceWarning because features are unscaled; the lab fixes `max_iter=1000`, so the configuration was kept as specified.
- Numbers were measured on Windows 10/11 with Python 3.11.8 and the pinned library versions in `results/versions.txt`. `pyarrow` is pinned to 15.0.2 instead of 16.1.0 because `mlflow==2.14.1` requires `pyarrow<16`; pyarrow is not used in this lab.
