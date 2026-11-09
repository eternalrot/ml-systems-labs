"""All Lab 6 studies. Every run is a separate process (clean peak-memory numbers).

    python src/studies.py all            # everything (long)
    python src/studies.py workers|batch|schedule|accum|resume|profile|export
Finished runs are cached in results/runs/<name>/metrics.json; use --force to repeat.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT / "results"
FORCE = False
TRAIN_SIZE = 20000        # studies use a 20 000-image training subset so that all runs finish in reasonable time


def run(name, overrides, epochs):
    """Run train.py as a subprocess (or reuse the cached result)."""
    mfile = RES / "runs" / name / "metrics.json"
    if mfile.exists() and not FORCE:
        return json.loads(mfile.read_text())
    cmd = [sys.executable, str(HERE / "train.py"), "--name", name, "--set", f"train.epochs={epochs}", f"data.train_size={TRAIN_SIZE}", *overrides]
    print(">>", " ".join(cmd[2:]), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)
    return json.loads(mfile.read_text())


def write_csv(name, header, rows):
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(x, n=4):
    return f"{x:.{n}f}"


# ------------------------------------------------------------------ 2. profiling
def study_profile():
    rows, out = [], {}
    for w in (0, 2):
        m = run(f"profile_w{w}", [f"data.num_workers={w}", "profile.enabled=true"], 1)
        p = m["profile"]
        out[w] = p
        for phase in ("data_loading", "forward", "backward", "optimizer_step"):
            rows.append([w, phase, fmt(p["profiler_seconds_recorded_steps"][phase], 3), fmt(100 * p["profiler_share"][phase], 1),
                         fmt(100 * p["timer_share"][phase], 1)])
    write_csv("profile_breakdown.csv", ["num_workers", "phase", "profiler_seconds_recorded_steps", "profiler_share_%",
                                        "wall_clock_timer_share_%"], rows)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    phases = ["data_loading", "forward", "backward", "optimizer_step"]
    for k, w in enumerate((0, 2)):
        ax.bar([i + 0.4 * k for i in range(4)], [100 * out[w]["profiler_share"][p] for p in phases], 0.4,
               label=f"num_workers={w}")
    ax.set_xticks([i + 0.2 for i in range(4)])
    ax.set_xticklabels(phases)
    ax.set_ylabel("share of one epoch (%)")
    ax.legend()
    ax.set_title("torch.profiler: where one epoch goes")
    fig.tight_layout()
    fig.savefig(RES / "profile_breakdown.png", dpi=130)


# ------------------------------------------------------------------ 3. workers
def study_workers():
    rows, xs, th, wait = [], [], [], []
    ref = run("workers_compute_only", ["data.num_workers=0", "data.augment=false", "data.preload=true"], 3)
    ref_th = ref["median_samples_per_s"]
    for w in (0, 1, 2, 4, 8):
        m = run(f"workers_{w}", [f"data.num_workers={w}"], 3)
        h = m["history"][1:]                                   # epochs 2-3: workers already started
        share = statistics.median(e["t_data_loading"] / e["epoch_seconds"] for e in h)
        rows.append([w, fmt(m["first_epoch_samples_per_s"], 0), fmt(m["median_samples_per_s"], 0),
                     fmt(100 * share, 1), fmt(m["median_samples_per_s"] / ref_th, 2)])
        xs.append(w); th.append(m["median_samples_per_s"]); wait.append(100 * share)
    rows.append(["0 (preloaded tensors, no augmentation = compute only)", fmt(ref["first_epoch_samples_per_s"], 0), fmt(ref_th, 0),
                 fmt(100 * statistics.median(e["t_data_loading"] / e["epoch_seconds"] for e in ref["history"][1:]), 1), "1.00"])
    write_csv("workers.csv", ["num_workers", "samples_per_s_epoch1", "samples_per_s_median_epoch2-3",
                              "time_waiting_for_data_%", "fraction_of_compute_only_speed"], rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(xs, th, "o-", label="with augmentation")
    ax[0].axhline(ref_th, ls="--", color="gray", label="compute only (preloaded tensors)")
    ax[0].set_xlabel("num_workers"); ax[0].set_ylabel("training throughput (samples/s)"); ax[0].legend(fontsize=8)
    ax[1].plot(xs, wait, "o-", color="C3")
    ax[1].set_xlabel("num_workers"); ax[1].set_ylabel("main process waiting for data (% of epoch)")
    fig.suptitle(f"DataLoader workers ({os.cpu_count()} logical CPUs, {torch.get_num_threads()} torch threads)")
    fig.tight_layout()
    fig.savefig(RES / "workers.png", dpi=130)


# ------------------------------------------------------------------ 4. batch size
def study_batch():
    rows, xs, th, mem, acc = [], [], [], [], []
    for b in (32, 64, 128, 256, 512):
        m = run(f"batch_{b}", [f"train.batch_size={b}"], 5)
        rows.append([b, fmt(m["median_samples_per_s"], 0), fmt(m["peak_rss_mb"], 0),
                     fmt(m["history"][-1]["val_acc"]), fmt(m["test_acc"]), fmt(m["history"][-1]["train_loss"])])
        xs.append(b); th.append(m["median_samples_per_s"]); mem.append(m["peak_rss_mb"]); acc.append(m["test_acc"])
    write_csv("batch_size.csv", ["batch_size", "throughput_samples_per_s", "peak_rss_mb(main process)",
                                 "val_acc_after_5_epochs", "test_acc_after_5_epochs", "final_train_loss"], rows)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for a, y, t in zip(ax, (th, mem, acc), ("throughput (samples/s)", "peak RSS (MB)", "test accuracy after 5 epochs")):
        a.plot(xs, y, "o-"); a.set_xscale("log", base=2); a.set_xlabel("batch size"); a.set_title(t)
    fig.tight_layout()
    fig.savefig(RES / "batch_size.png", dpi=130)


# ------------------------------------------------------------------ 5. schedules and optimizers
def study_schedule():
    cfgs = [
        ("sched_none", "Adam, constant lr", ["train.scheduler.name=none"]),
        ("sched_step", "Adam + StepLR", ["train.scheduler.name=step"]),
        ("sched_cosine", "Adam + CosineAnnealingLR", ["train.scheduler.name=cosine"]),
        ("sched_plateau", "Adam + ReduceLROnPlateau (patience 0)", ["train.scheduler.name=plateau", "train.scheduler.patience=0"]),
        ("opt_sgd_cosine", "SGD+momentum + CosineAnnealingLR",
         ["train.scheduler.name=cosine", "train.optimizer.name=sgd", "train.optimizer.lr=0.05"]),
    ]
    rows, res = [], {}
    for name, label, ov in cfgs:
        m = run(name, ov, 10)
        res[name] = (label, m)
        rows.append([label, fmt(m["history"][-1]["lr"], 5), fmt(m["best_val_acc"]), fmt(m["history"][-1]["val_acc"]),
                     fmt(m["test_acc"]), fmt(m["history"][-1]["train_loss"])])
    write_csv("schedules_optimizers.csv", ["configuration", "last_epoch_lr", "best_val_acc", "final_val_acc",
                                           "test_acc", "final_train_loss"], rows)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for name, (label, m) in res.items():
        e = [h["epoch"] for h in m["history"]]
        ax[0].plot(e, [h["lr"] for h in m["history"]], label=label)
        ax[1].plot(e, [h["val_acc"] for h in m["history"]], label=label)
        ax[2].plot(e, [h["train_loss"] for h in m["history"]], label=label)
    for a, t in zip(ax, ("learning rate", "validation accuracy", "training loss")):
        a.set_xlabel("epoch"); a.set_title(t)
    ax[0].set_yscale("log"); ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RES / "schedules_optimizers.png", dpi=130)


# ------------------------------------------------------------------ 6. gradient accumulation
def grad_equivalence():
    """Same 256 samples: one batch of 256 vs 4 micro-batches of 64 accumulated. Gradients must agree."""
    sys.path.insert(0, str(HERE))
    from models import CNN
    torch.manual_seed(42)
    x, y = torch.randn(256, 1, 28, 28), torch.randint(0, 10, (256,))
    m1 = CNN(); m2 = CNN(); m2.load_state_dict(m1.state_dict())
    lf = torch.nn.CrossEntropyLoss()
    lf(m1(x), y).backward()
    for xc, yc in zip(x.chunk(4), y.chunk(4)):
        (lf(m2(xc), yc) / 4).backward()
    d = max((a.grad - b.grad).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters()))
    return d


def study_accum():
    d = grad_equivalence()
    cfgs = [("accum_b256", "batch 256", ["train.batch_size=256", "train.accum_steps=1"]),
            ("accum_b64x4", "batch 64 x accumulation 4", ["train.batch_size=64", "train.accum_steps=4"]),
            ("accum_b64", "batch 64 (no accumulation)", ["train.batch_size=64", "train.accum_steps=1"])]
    rows, res = [], {}
    for name, label, ov in cfgs:
        m = run(name, ov, 5)
        res[name] = (label, m)
        rows.append([label, fmt(m["history"][-1]["train_loss"]), fmt(m["history"][-1]["val_acc"]), fmt(m["test_acc"]),
                     fmt(m["median_samples_per_s"], 0), fmt(m["peak_rss_mb"], 0)])
    write_csv("grad_accumulation.csv", ["configuration", "final_train_loss", "val_acc", "test_acc",
                                        "samples_per_s", "peak_rss_mb"], rows)
    write_csv("grad_accumulation_check.csv", ["check", "value"],
              [["max abs gradient difference, batch 256 vs 4 x 64 accumulated (one step, float32)", d]])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, (label, m) in res.items():
        e = [h["epoch"] for h in m["history"]]
        ax[0].plot(e, [h["train_loss"] for h in m["history"]], "o-", label=label)
        ax[1].plot(e, [h["val_acc"] for h in m["history"]], "o-", label=label)
    ax[0].set_title("training loss"); ax[1].set_title("validation accuracy"); ax[1].legend(fontsize=8)
    for a in ax: a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(RES / "grad_accumulation.png", dpi=130)


# ------------------------------------------------------------------ 7. interrupt and resume
def study_resume():
    base = ["data.num_workers=0"]                       # single process: random state is fully restored
    full = run("resume_full", base, 6)
    p1 = run("resume_part1", base + ["train.stop_after_epoch=3"], 6)
    p2 = run("resume_part2", base + ["checkpoint.resume_from=checkpoints/resume_part1/last.pt"], 6)
    joined = p2["history"]                              # the history stored in the checkpoint + epochs 4-6
    rows, md, ma = [], 0.0, 0.0
    for a, b in zip(full["history"], joined):
        dl, da = abs(a["train_loss"] - b["train_loss"]), abs(a["val_acc"] - b["val_acc"])
        md, ma = max(md, dl), max(ma, da)
        rows.append([a["epoch"], fmt(a["train_loss"], 6), fmt(b["train_loss"], 6), fmt(a["val_acc"], 4), fmt(b["val_acc"], 4)])
    write_csv("resume.csv", ["epoch", "train_loss_uninterrupted", "train_loss_interrupted+resumed",
                             "val_acc_uninterrupted", "val_acc_interrupted+resumed"], rows)
    write_csv("resume_check.csv", ["quantity", "value"],
              [["epochs completed before interruption", p1["epochs_done"]],
               ["max |train loss difference|", md], ["max |val acc difference|", ma],
               ["final test acc uninterrupted", full["test_acc"]], ["final test acc resumed", p2["test_acc"]]])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    e = [h["epoch"] for h in full["history"]]
    for a, key in zip(ax, ("train_loss", "val_acc")):
        a.plot(e, [h[key] for h in full["history"]], "o-", label="uninterrupted")
        a.plot(e, [h[key] for h in joined], "x--", label="interrupted after epoch 3, resumed")
        a.axvline(3, color="gray", ls=":"); a.set_xlabel("epoch"); a.set_title(key)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "resume.png", dpi=130)


# ------------------------------------------------------------------ MLflow export
def export_mlflow():
    import mlflow
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    df = mlflow.search_runs(experiment_names=["lab6"])
    keep = [c for c in df.columns if c.startswith(("tags.run_name", "metrics.test_acc", "metrics.median_samples_per_s",
                                                   "metrics.peak_rss_mb", "params.train.", "params.data.num_workers",
                                                   "params.model.name", "status", "run_id"))]
    df[keep].sort_values("tags.run_name").to_csv(RES / "mlflow_runs.csv", index=False)
    print(f"exported {len(df)} MLflow runs")


STUDIES = {"profile": study_profile, "workers": study_workers, "batch": study_batch, "schedule": study_schedule,
           "accum": study_accum, "resume": study_resume}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=[*STUDIES, "all", "export"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    FORCE = a.force
    RES.mkdir(exist_ok=True)
    if a.what == "export":
        export_mlflow()
    else:
        for n in (STUDIES if a.what == "all" else [a.what]):
            print(f"===== {n} =====", flush=True)
            STUDIES[n]()
        export_mlflow()
