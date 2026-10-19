"""Lab 3 driver. Run from Lab3/: python src/lab3.py"""
import csv
import os
import platform
import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from thop import profile as thop_profile
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary
from torchvision import datasets, transforms

from flops import model_flops, one_layer_examples
from models import build_models

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "..", "data")
os.makedirs(RES, exist_ok=True)


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def write_csv(name, header, rows):
    with open(os.path.join(RES, name), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


seed_all()
print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, {platform.processor() or platform.machine()}")

# ---- Step 1: data --------------------------------------------------------
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
train_full = datasets.FashionMNIST(root=DATA, train=True, download=True, transform=tf)
test_set = datasets.FashionMNIST(root=DATA, train=False, download=True, transform=tf)


def to_tensors(ds):
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    return TensorDataset(x, y)


train_t = to_tensors(train_full)
test_t = to_tensors(test_set)
train_set, val_set = random_split(train_t, [55000, 5000], generator=torch.Generator().manual_seed(SEED))
print(f"train {len(train_set)}, val {len(val_set)}, test {len(test_t)}")
BS = 128
train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, generator=torch.Generator().manual_seed(SEED))
val_loader = DataLoader(val_set, batch_size=1000)
test_loader = DataLoader(test_t, batch_size=1000)

# ---- Step 2: models ------------------------------------------------------
models = {}
for name in ("MLP", "CNN", "Attention"):
    seed_all()                      # same seed before each model is built
    models[name] = build_models()[name]
for n, m in models.items():
    print(n, sum(p.numel() for p in m.parameters()), "parameters")

# ---- Step 3a: parameters (torchinfo) --------------------------------------
param_rows = []
for n, m in models.items():
    s = summary(m, input_size=(1, 1, 28, 28), verbose=0, depth=4,
                col_names=("input_size", "output_size", "num_params", "mult_adds"))
    with open(os.path.join(RES, f"torchinfo_{n}.txt"), "w", encoding="utf-8") as f:
        f.write(str(s))
    manual = sum(p.numel() for p in m.parameters())
    param_rows.append([n, s.total_params, manual, s.total_params == manual])
write_csv("params.csv", ["model", "torchinfo_total_params", "sum_of_param_numel", "match"], param_rows)
params = {r[0]: r[1] for r in param_rows}

# ---- Step 3b: FLOPs: hand vs thop vs torch FlopCounter ----------------------
hand = model_flops()
write_csv("flops_one_layer_hand.csv", ["layer", "flops"], list(one_layer_examples().items()))
x1 = torch.randn(1, 1, 28, 28)


def thop_flops(m):
    macs, _ = thop_profile(m, inputs=(x1,), verbose=False)
    return 2 * macs                 # thop counts multiply-accumulates; 1 MAC = 2 FLOPs


def counter_flops(m):
    m.eval()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x1)
    return fc.get_total_flops()


flop_rows = []
for n, m in models.items():
    m.eval()
    torch.backends.mha.set_fastpath_enabled(True)
    th_fast = thop_flops(m)
    torch.backends.mha.set_fastpath_enabled(False)   # fused fast path hides sub-layers from hooks
    th_slow = thop_flops(m)
    fc = counter_flops(m)
    torch.backends.mha.set_fastpath_enabled(True)
    flop_rows.append([n, hand[n], th_fast, th_slow, fc])
    print(f"{n}: hand {hand[n]:,}  thop(default) {th_fast:,}  thop(no fast path) {th_slow:,}  torch FlopCounter {fc:,}")
write_csv("flops_compare.csv",
          ["model", "hand_flops", "thop_flops_default", "thop_flops_fastpath_off", "torch_flop_counter"], flop_rows)

# ---- Step 3c: latency ------------------------------------------------------
def latency(m, batch, runs, warmup):
    m.eval()
    x = torch.randn(batch, 1, 28, 28)
    ts = []
    with torch.no_grad():
        for _ in range(warmup):
            m(x)
        for _ in range(runs):
            t0 = time.perf_counter()
            m(x)
            ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


lat = {}
for n, m in models.items():
    lat[n] = (latency(m, 1, 100, 20), latency(m, 128, 20, 5))
write_csv("latency.csv", ["model", "batch1_median_ms(100 runs)", "batch128_median_ms(20 runs)",
                          "batch128_per_sample_us"],
          [[n, round(a, 4), round(b, 3), round(b / 128 * 1000, 2)] for n, (a, b) in lat.items()])
for n, (a, b) in lat.items():
    print(f"{n}: latency batch1 {a:.3f} ms, batch128 {b:.2f} ms")

# ---- Step 4: training --------------------------------------------------------
@torch.no_grad()
def evaluate(m, loader):
    m.eval()
    correct = 0
    for x, y in loader:
        correct += (m(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


log_rows, acc = [], {}
for n, m in models.items():
    seed_all()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(1, 6):
        m.train()
        tot, cnt = 0.0, 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            opt.zero_grad()
            loss = lossf(m(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            cnt += len(y)
        va = evaluate(m, val_loader)
        log_rows.append([n, ep, round(tot / cnt, 4), round(va, 4), round(time.perf_counter() - t0, 1)])
        print(f"{n} epoch {ep}: train loss {tot/cnt:.4f}  val acc {va:.4f}")
    acc[n] = evaluate(m, test_loader)
    print(f"{n}: TEST accuracy {acc[n]:.4f}")
write_csv("training_log.csv", ["model", "epoch", "train_loss", "val_accuracy", "epoch_seconds"], log_rows)
write_csv("test_accuracy.csv", ["model", "test_accuracy"], [[n, round(a, 4)] for n, a in acc.items()])

# ---- summary + Step 5: trade-off plots -----------------------------------------
write_csv("summary.csv",
          ["model", "params", "hand_flops", "test_accuracy", "latency_b1_ms", "latency_b128_ms"],
          [[n, params[n], hand[n], round(acc[n], 4), round(lat[n][0], 4), round(lat[n][1], 3)] for n in models])

colors = {"MLP": "tab:blue", "CNN": "tab:green", "Attention": "tab:red"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
panels = [("Parameters", lambda n: params[n], True), ("Latency, batch 1 (ms)", lambda n: lat[n][0], False),
          ("Latency, batch 128 (ms)", lambda n: lat[n][1], False)]
for ax, (label, f, _) in zip(axes, panels):
    for n in models:
        ax.scatter(f(n), acc[n], s=120, color=colors[n], label=n, zorder=3)
        ax.annotate(f"{acc[n]:.3f}", (f(n), acc[n]), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel(label)
    ax.set_ylabel("test accuracy")
    ax.grid(alpha=.3)
axes[0].legend()
plt.suptitle("Fashion-MNIST: accuracy vs cost (5 epochs, Adam 1e-3)")
plt.tight_layout()
plt.savefig(os.path.join(RES, "tradeoff_plots.png"), dpi=150)
print("done")
