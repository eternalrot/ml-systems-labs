"""Project milestone: trained baseline model on the data produced by the Lab 4 pipeline, and metrics.json.

    cd Project
    python data_pipeline/src/run_pipeline.py      # data pipeline (Lab 4), writes data/processed/*.csv
    python src/train_baseline.py                  # baseline model -> models/baseline.pt and metrics.json
"""
import io
import json
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
PIPE = ROOT / "data_pipeline"
SEED = 42
EPOCHS, BATCH, LR = 20, 256, 1e-3
LIMITS = {"latency_ms": 50.0, "size_mb": 5.0}          # deployment constraints of the project proposal


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(n_in):
    return nn.Sequential(nn.Linear(n_in, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))


def main():
    seed_all(SEED)
    tr, te = pd.read_csv(PROC / "adult_train_processed.csv"), pd.read_csv(PROC / "adult_test_processed.csv")
    feats = [c for c in tr.columns if c != "income"]
    X, y = tr[feats].values.astype(np.float32), tr["income"].values.astype(np.float32)
    Xte, yte = te[feats].values.astype(np.float32), te["income"].values.astype(np.float32)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=y)
    Xtr, Xva, ytr, yva, Xte_t = map(torch.tensor, (Xtr, Xva, ytr, yva, Xte))

    model = make_model(len(feats))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(SEED)
    best, best_state, t0 = (1e9, 0), None, time.perf_counter()
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model(Xtr[idx]).squeeze(1), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(Xva).squeeze(1), yva).item()
        if vl < best[0]:
            best, best_state = (vl, ep), {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:2d} val loss {vl:.4f}", flush=True)
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)                    # keep the epoch with the lowest validation loss
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte_t).squeeze(1)).numpy()
    pred = (p > 0.5).astype(int)

    # size and latency against the proposal constraints
    buf = io.BytesIO(); torch.save(model.state_dict(), buf)
    torch.set_num_threads(1)
    x1 = Xte_t[:1]
    ts = []
    with torch.no_grad():
        for _ in range(50): model(x1)
        for _ in range(300):
            t = time.perf_counter(); model(x1); ts.append((time.perf_counter() - t) * 1e3)
    lat = statistics.median(ts)

    (ROOT / "models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "models" / "baseline.pt")
    manifest = {}
    mf = PIPE / "results" / "run_manifest.yaml"
    if mf.exists():
        m = yaml.safe_load(open(mf))
        manifest = {"sha256": m.get("sha256"), "shapes": m.get("shapes"), "ingestion_source": m.get("ingestion", {}).get("source")}
    ref = None
    bm = PIPE / "results" / "baseline_metrics.csv"
    if bm.exists():
        r = pd.read_csv(bm)
        r = r[r["derived_features"] == True].iloc[0]
        ref = {"model": "LogisticRegression (Lab 4)", "accuracy": float(r["accuracy"]), "f1": float(r["f1_income_gt_50K"])}
    metrics = {
        "project": "Adult income (>50K) classifier, baseline milestone",
        "data": {"train_rows": int(len(tr)), "test_rows": int(len(te)), "features": len(feats),
                 "positive_rate_test": round(float(yte.mean()), 4), **manifest},
        "model": {"name": "MLP 107-128-64-1", "parameters": int(sum(p_.numel() for p_ in model.parameters())),
                  "file": "models/baseline.pt", "best_epoch": best[1], "epochs": EPOCHS, "batch_size": BATCH,
                  "optimizer": f"Adam lr={LR}", "seed": SEED, "train_seconds": round(train_seconds, 1)},
        "test_metrics": {"accuracy": round(float(accuracy_score(yte, pred)), 4), "f1_income_gt_50K": round(float(f1_score(yte, pred)), 4),
                         "roc_auc": round(float(roc_auc_score(yte, p)), 4)},
        "reference_baseline": ref,
        "deployment": {"model_size_kb": round(len(buf.getvalue()) / 1024, 1), "latency_batch1_ms_1thread": round(lat, 4),
                       "limits": LIMITS,
                       "meets_latency": bool(lat <= LIMITS["latency_ms"]),
                       "meets_size": bool(len(buf.getvalue()) / 2 ** 20 <= LIMITS["size_mb"])},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform()},
    }
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics["test_metrics"], indent=2), metrics["deployment"])


if __name__ == "__main__":
    main()
