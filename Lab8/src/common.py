"""Data, training, latency and size helpers shared by the Lab 8 steps (same recipe as Lab 7)."""
import io
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

ROOT = Path(__file__).resolve().parent.parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(cfg):
    root = str((ROOT / cfg["data"]["root"]).resolve())
    tr = datasets.FashionMNIST(root=root, train=True, download=True)
    te = datasets.FashionMNIST(root=root, train=False, download=True)

    def prep(ds):
        return ((ds.data.float() / 255 - 0.2860) / 0.3530).unsqueeze(1), ds.targets.clone()
    X, y = prep(tr)
    Xte, yte = prep(te)
    g = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(X), generator=g)
    v = cfg["data"]["val_size"]
    return (X[perm[v:]], y[perm[v:]]), (X[perm[:v]], y[perm[:v]]), (Xte, yte)


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return sum((model(X[i:i + 1000]).argmax(1) == y[i:i + 1000]).sum().item() for i in range(0, len(X), 1000)) / len(X)


def train(model, data, epochs, lr, seed, bs=128, masks=None, teacher_logits=None, T=4.0, alpha=0.5, log=None):
    X, y = data
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), bs):
            idx = perm[i:i + bs]
            out = model(X[idx])
            loss = F.cross_entropy(out, y[idx])
            if teacher_logits is not None:
                kl = F.kl_div(F.log_softmax(out / T, 1), F.softmax(teacher_logits[idx] / T, 1), reduction="batchmean")
                loss = alpha * loss + (1 - alpha) * T * T * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if masks is not None:
                for n, w in masks.items():
                    getattr(model, n).weight.data.mul_(w)
            tot += loss.item() * len(idx)
        if log:
            log(ep, tot / len(X))
    return time.perf_counter() - t0


@torch.no_grad()
def latency_ms(fn, make_input, batch, runs, warmup, threads=None):
    """Median latency of fn(input) in ms. fn is any callable (PyTorch module or ONNX Runtime wrapper)."""
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    x = make_input(batch)
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev)
    return statistics.median(ts)


def img_batch(batch):
    return torch.randn(batch, 1, 28, 28, generator=torch.Generator().manual_seed(0))


def measure_latency(fn, cfg, make_input=img_batch):
    L = cfg["latency"]
    return (latency_ms(fn, make_input, 1, L["batch1_runs"], L["warmup"], threads=1),
            latency_ms(fn, make_input, 128, L["batch128_runs"], L["warmup"]))


def state_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue())
