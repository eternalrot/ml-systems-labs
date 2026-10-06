"""Lab 1: baselines + system cost measurements. Run from lab01/: python src/measure.py"""
import csv
import os
import random
import statistics
import threading
import time

import joblib
import numpy as np
import psutil
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
PROC = psutil.Process()


def make_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    }


class PeakRSS:
    """Samples process RSS in a background thread; reports peak MB above baseline."""

    def __enter__(self):
        self.base = PROC.memory_info().rss
        self.peak = self.base
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, PROC.memory_info().rss)
            time.sleep(0.001)

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        self.peak = max(self.peak, PROC.memory_info().rss)

    @property
    def peak_mb(self):
        return self.peak / 1e6

    @property
    def delta_mb(self):
        return (self.peak - self.base) / 1e6


def main():
    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)

    rows = []
    for name, _ in make_models().items():
        # --- accuracy -------------------------------------------------------
        model = make_models()[name]
        model.fit(X_tr, y_tr)
        acc = round(accuracy_score(y_te, model.predict(X_te)), 4)

        # --- training time: 1 warm-up, median of 5 --------------------------
        make_models()[name].fit(X_tr, y_tr)  # warm-up
        times = []
        for _ in range(5):
            m = make_models()[name]
            t0 = time.perf_counter()
            m.fit(X_tr, y_tr)
            times.append(time.perf_counter() - t0)
        train_ms = statistics.median(times) * 1000

        # --- single-sample inference: 100 repeats, median -------------------
        sample = X_te[:1]
        model.predict(sample)  # warm-up
        lat = []
        for _ in range(100):
            t0 = time.perf_counter()
            model.predict(sample)
            lat.append((time.perf_counter() - t0) * 1000)
        infer_ms = statistics.median(lat)

        # --- model size -----------------------------------------------------
        path = os.path.join(RESULTS, f"model_{name}.joblib")
        joblib.dump(model, path)
        size_b = os.path.getsize(path)

        # --- peak memory (RSS) during training and inference ----------------
        with PeakRSS() as mt:
            make_models()[name].fit(X_tr, y_tr)
        with PeakRSS() as mi:
            for _ in range(100):
                model.predict(sample)

        rows.append(dict(
            model=name, test_accuracy=acc,
            train_median_ms=round(train_ms, 3),
            infer_single_median_ms=round(infer_ms, 4),
            model_size_bytes=size_b, model_size_kb=round(size_b / 1024, 2),
            peak_rss_train_mb=round(mt.peak_mb, 1),
            peak_rss_infer_mb=round(mi.peak_mb, 1),
            delta_rss_train_mb=round(mt.delta_mb, 2),
            delta_rss_infer_mb=round(mi.delta_mb, 2),
        ))

    # Required file names: model.joblib (the RF, the larger one) + per-model files
    joblib.dump(joblib.load(os.path.join(RESULTS, "model_RandomForest.joblib")),
                os.path.join(RESULTS, "model.joblib"))

    with open(os.path.join(RESULTS, "baseline_accuracy.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "test_accuracy"])
        for r in rows:
            w.writerow([r["model"], f"{r['test_accuracy']:.4f}"])

    with open(os.path.join(RESULTS, "system_cost.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # --- budget table -------------------------------------------------------
    budgets = {  # mem range MB, latency ms, size KB
        "Cloud":  dict(mem=(1024, float("inf")), lat=100, size=500 * 1024),
        "Edge":   dict(mem=(256, 1024), lat=50, size=50 * 1024),
        "Mobile": dict(mem=(64, 256), lat=20, size=10 * 1024),
        "TinyML": dict(mem=(0, 0.256), lat=10, size=100),
    }
    with open(os.path.join(RESULTS, "budget_fit.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "target", "latency_ok", "size_ok", "memory_ok", "fits"])
        for r in rows:
            for tgt, b in budgets.items():
                lat_ok = r["infer_single_median_ms"] <= b["lat"]
                size_ok = r["model_size_kb"] <= b["size"]
                # Memory: runtime RSS of the whole Python process. For TinyML the
                # limit is an upper bound (<=256 KB); for the others the target
                # device has that much memory, so the process must fit within the
                # upper end of the range.
                limit = b["mem"][1] if tgt != "TinyML" else 0.256
                mem_ok = r["peak_rss_infer_mb"] <= limit
                w.writerow([r["model"], tgt, lat_ok, size_ok, mem_ok,
                            lat_ok and size_ok and mem_ok])

    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
