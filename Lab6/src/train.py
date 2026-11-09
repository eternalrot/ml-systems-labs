"""YAML-driven training pipeline with checkpoints, resume, MLflow logging and optional profiling.

    python src/train.py --config config.yaml --name my_run --set train.epochs=3 data.num_workers=0
"""
import os

os.environ.setdefault("KINETO_LOG_LEVEL", "5")      # silence profiler stage messages

import argparse
import csv
import json
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # Lab6/
sys.path.insert(0, str(HERE))
from data import make_loaders           # noqa: E402
from models import MODELS               # noqa: E402


# ------------------------------------------------------------------ config
def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key: {key}")
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ------------------------------------------------------------------ memory
class PeakRSS(threading.Thread):
    """Polls the resident memory of the training process and keeps the maximum."""

    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.proc, self.interval, self.peak, self._stop_flag = psutil.Process(), interval, 0, threading.Event()

    def sample(self):
        try:
            self.peak = max(self.peak, self.proc.memory_info().rss)    # main process only (workers share pages)
        except psutil.Error:
            pass

    def run(self):
        while not self._stop_flag.is_set():
            self.sample()
            time.sleep(self.interval)

    def stop(self):
        self._stop_flag.set()
        self.sample()
        return self.peak / 2 ** 20


# ------------------------------------------------------------------ build
def build_optimizer(model, o):
    if o["name"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    if o["name"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=o["lr"], momentum=o["momentum"], weight_decay=o["weight_decay"])
    raise ValueError(o["name"])


def build_scheduler(opt, s, epochs):
    n = s["name"]
    if n == "none":
        return None
    if n == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=s["step_size"], gamma=s["gamma"])
    if n == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if n == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=s["factor"], patience=s["patience"])
    raise ValueError(n)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    tot, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot += loss_fn(out, y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        n += len(y)
    return tot / n, correct / n


# ------------------------------------------------------------------ one epoch
def train_one_epoch(model, loader, opt, loss_fn, accum, device, prof=None):
    """Returns (mean loss, seconds, dict of time per phase). Phases are labelled with record_function
    so that torch.profiler can attribute time to them as well."""
    model.train()
    t = {"data_loading": 0.0, "forward": 0.0, "backward": 0.0, "optimizer_step": 0.0}
    it = iter(loader)
    total_loss, n_samples, i = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    start = time.perf_counter()
    while True:
        t0 = time.perf_counter()
        with record_function("data_loading"):
            try:
                x, y = next(it)
            except StopIteration:
                break
            x, y = x.to(device), y.to(device)
        t1 = time.perf_counter()
        with record_function("forward"):
            out = model(x)
            loss = loss_fn(out, y)
        t2 = time.perf_counter()
        with record_function("backward"):
            (loss / accum).backward()           # accumulate gradients of `accum` batches
        t3 = time.perf_counter()
        i += 1
        with record_function("optimizer_step"):
            if i % accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
        t4 = time.perf_counter()
        if prof is not None:
            prof.step()
        t["data_loading"] += t1 - t0
        t["forward"] += t2 - t1
        t["backward"] += t3 - t2
        t["optimizer_step"] += t4 - t3
        total_loss += loss.item() * len(y)
        n_samples += len(y)
    if i % accum != 0:                          # leftover micro-batches at the end of the epoch
        opt.step()
        opt.zero_grad(set_to_none=True)
    return total_loss / n_samples, time.perf_counter() - start, t, n_samples


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path, model, opt, sched, epoch, history, best_val, gen, run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "epoch": epoch, "history": history, "best_val": best_val, "mlflow_run_id": run_id,
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "loader": gen.get_state()},
    }, path)


# ------------------------------------------------------------------ main run
def run(cfg, name):
    seed_all(cfg["seed"])
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "results" / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = ROOT / cfg["checkpoint"]["dir"] / name
    t_cfg = cfg["train"]

    mem = PeakRSS()
    base_rss = psutil.Process().memory_info().rss / 2 ** 20
    mem.start()

    gen = torch.Generator().manual_seed(cfg["seed"])
    train_loader, val_loader, test_loader = make_loaders(cfg, ROOT, gen)
    model = MODELS[cfg["model"]["name"]]().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = build_optimizer(model, t_cfg["optimizer"])
    sched = build_scheduler(opt, t_cfg["scheduler"], t_cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss()

    history, start_epoch, best_val, run_id = [], 0, 0.0, None
    if cfg["checkpoint"]["resume_from"]:
        ck = torch.load(ROOT / cfg["checkpoint"]["resume_from"], weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if sched and ck["scheduler"]:
            sched.load_state_dict(ck["scheduler"])
        history, start_epoch, best_val, run_id = ck["history"], ck["epoch"], ck["best_val"], ck["mlflow_run_id"]
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        gen.set_state(ck["rng"]["loader"])
        print(f"[{name}] resumed from epoch {start_epoch}")

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    status = "complete"
    prof_summary = None
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else name) as active:
        run_id = active.info.run_id
        if not cfg["checkpoint"]["resume_from"]:
            mlflow.log_params({k: str(v) for k, v in flatten(cfg).items()})
            mlflow.log_param("n_params", n_params)
        mlflow.set_tags({"run_name": name, "resumed": str(bool(cfg["checkpoint"]["resume_from"]))})

        train_start = time.perf_counter()
        for epoch in range(start_epoch + 1, t_cfg["epochs"] + 1):
            lr = opt.param_groups[0]["lr"]
            if cfg["profile"]["enabled"]:
                # The whole epoch is covered by a cycle of 15 skipped steps, 2 warm-up steps and 8 recorded steps,
                # so memory stays bounded; the recorded steps are summed up in on_ready().
                tot_us, self_us = {}, {}

                def on_ready(p):
                    for e in p.key_averages():
                        tot_us[e.key] = tot_us.get(e.key, 0.0) + e.cpu_time_total
                        self_us[e.key] = self_us.get(e.key, 0.0) + e.self_cpu_time_total

                with profile(activities=[ProfilerActivity.CPU], schedule=schedule(wait=15, warmup=2, active=8),
                             on_trace_ready=on_ready) as prof:
                    loss, secs, phases, n = train_one_epoch(model, train_loader, opt, loss_fn,
                                                            t_cfg["accum_steps"], device, prof)
                labelled = {k: tot_us.get(k, 0.0) / 1e6 for k in phases}
                s_prof = sum(labelled.values())
                prof_summary = {
                    "epoch_seconds": secs,
                    "timer_seconds": phases,
                    "timer_share": {k: v / sum(phases.values()) for k, v in phases.items()},
                    "profiler_seconds_recorded_steps": labelled,
                    "profiler_share": {k: v / s_prof for k, v in labelled.items()},
                }
                top = sorted(self_us.items(), key=lambda kv: -kv[1])[:15]
                total_self = sum(self_us.values())
                (out_dir / "profiler_table.txt").write_text(
                    "top operators by self CPU time over the recorded steps\n"
                    + "\n".join(f"{k[:60]:60s} {v / 1e3:10.1f} ms {100 * v / total_self:5.1f} %" for k, v in top) + "\n")
            else:
                loss, secs, phases, n = train_one_epoch(model, train_loader, opt, loss_fn, t_cfg["accum_steps"], device)
            val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
            if sched:
                sched.step(val_acc) if t_cfg["scheduler"]["name"] == "plateau" else sched.step()
            row = {"epoch": epoch, "lr": lr, "train_loss": loss, "val_loss": val_loss, "val_acc": val_acc,
                   "epoch_seconds": secs, "samples_per_s": n / secs,
                   **{f"t_{k}": v for k, v in phases.items()}}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            best_val = max(best_val, val_acc)
            save_checkpoint(ckpt_dir / "last.pt", model, opt, sched, epoch, history, best_val, gen, run_id)
            if val_acc >= best_val:
                save_checkpoint(ckpt_dir / "best.pt", model, opt, sched, epoch, history, best_val, gen, run_id)
            print(f"[{name}] epoch {epoch}/{t_cfg['epochs']} lr {lr:.5f} loss {loss:.4f} val_acc {val_acc:.4f} "
                  f"{n / secs:.0f} samples/s", flush=True)
            if t_cfg["stop_after_epoch"] and epoch == t_cfg["stop_after_epoch"]:
                status = "interrupted"
                print(f"[{name}] simulated interruption after epoch {epoch}")
                break
        train_seconds = time.perf_counter() - train_start

        test_acc = None
        if status == "complete":
            _, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
            mlflow.log_metric("test_acc", test_acc)
        peak = mem.stop()
        later = [h["samples_per_s"] for h in history[1:]] or [h["samples_per_s"] for h in history]
        summary = {
            "name": name, "status": status, "epochs_done": len(history), "n_params": n_params,
            "final_val_acc": history[-1]["val_acc"], "best_val_acc": best_val, "test_acc": test_acc,
            "median_samples_per_s": statistics.median(later),
            "first_epoch_samples_per_s": history[0]["samples_per_s"],
            "peak_rss_mb": peak, "rss_before_mb": base_rss, "train_seconds": train_seconds,
            "history": history, "config": cfg, "profile": prof_summary,
            "env": {"python": platform.python_version(), "torch": torch.__version__,
                    "threads": torch.get_num_threads(), "platform": platform.platform(),
                    "cpu_count": os.cpu_count()},
        }
        mlflow.log_metrics({"peak_rss_mb": peak, "median_samples_per_s": summary["median_samples_per_s"]})
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        with open(out_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)
        with open(out_dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        mlflow.log_artifact(str(out_dir / "config.yaml"))
        mlflow.log_artifact(str(out_dir / "metrics.json"))
        if cfg["mlflow"]["log_checkpoint"]:
            mlflow.log_artifact(str(ckpt_dir / "last.pt"))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--name", default="run")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.epochs=3")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    s = run(cfg, a.name)
    print(json.dumps({k: s[k] for k in ("name", "status", "final_val_acc", "test_acc", "median_samples_per_s",
                                        "peak_rss_mb")}, indent=2))


if __name__ == "__main__":
    main()
