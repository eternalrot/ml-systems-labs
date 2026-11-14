"""Lab 7 driver: baseline, pruning (global / layer-wise / structured), latency and size, distillation, summary.
Run from Lab7/:   python src/lab7.py            (about 10-15 minutes)
"""
import copy
import csv
import io
import json
import os
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import yaml
from torchvision import datasets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from models import baseline, student           # noqa: E402
import prune                                   # noqa: E402

RES = ROOT / "results"
CK = ROOT / "checkpoints"
CFG = yaml.safe_load(open(ROOT / "config.yaml"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_csv(name, header, rows):
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ------------------------------------------------------------------ data
def load_data():
    root = str((ROOT / CFG["data"]["root"]).resolve())
    tr = datasets.FashionMNIST(root=root, train=True, download=True)
    te = datasets.FashionMNIST(root=root, train=False, download=True)

    def prep(ds):
        return ((ds.data.float() / 255 - 0.2860) / 0.3530).unsqueeze(1), ds.targets.clone()
    X, y = prep(tr)
    Xte, yte = prep(te)
    g = torch.Generator().manual_seed(CFG["seed"])
    perm = torch.randperm(len(X), generator=g)
    v = CFG["data"]["val_size"]
    return (X[perm[v:]], y[perm[v:]]), (X[perm[:v]], y[perm[:v]]), (Xte, yte)


@torch.no_grad()
def accuracy(model, X, y):
    model.eval()
    return sum((model(X[i:i + 2000]).argmax(1) == y[i:i + 2000]).sum().item() for i in range(0, len(X), 2000)) / len(X)


def train(model, data, epochs, lr, seed, masks=None, teacher_logits=None, T=4.0, alpha=0.5, log=None):
    """Adam training. masks: re-applied after every step (pruned weights stay zero). teacher_logits: logits of the
    teacher for every training image -> distillation loss alpha*CE + (1-alpha)*T^2*KL."""
    X, y = data
    bs = CFG["train"]["batch_size"]
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
                prune.apply_masks(model, masks)
            tot += loss.item() * len(idx)
        if log is not None:
            log(ep, tot / len(X))
    return time.perf_counter() - t0


# ------------------------------------------------------------------ latency, size
@torch.no_grad()
def latency_ms(model, batch, runs, threads=None):
    model.eval()
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    x = torch.randn(batch, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    for _ in range(CFG["latency"]["warmup"]):
        model(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    torch.set_num_threads(prev)
    return statistics.median(ts)


def measure_latency(model):
    return (latency_ms(model, 1, CFG["latency"]["batch1_runs"], threads=1),
            latency_ms(model, 128, CFG["latency"]["batch128_runs"]))


def dense_bytes(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue())


def csr_bytes(model):
    """Weights stored as CSR (float32 values, int32 column indices and row pointers); biases dense."""
    buf = io.BytesIO()
    arrays = {}
    for n, layer in model.layers().items():
        m = sp.csr_matrix(layer.weight.detach().numpy().reshape(layer.weight.shape[0], -1))
        arrays[n + "_data"] = m.data.astype(np.float32)
        arrays[n + "_indices"] = m.indices.astype(np.int32)
        arrays[n + "_indptr"] = m.indptr.astype(np.int32)
        arrays[n + "_bias"] = layer.bias.detach().numpy()
    np.savez(buf, **arrays)
    return len(buf.getvalue())


def describe(model, data_te, ft_seconds=0.0):
    sparsity, nnz, total = prune.sparsity_of(model)
    l1, l128 = measure_latency(model)
    return {
        "test_acc": accuracy(model, *data_te),
        "params": sum(p.numel() for p in model.parameters()),
        "weights_nonzero": nnz, "weights_total": total, "weight_sparsity": sparsity,
        "macs_dense": model.macs(False), "macs_nonzero": model.macs(True),
        "size_dense_kb": dense_bytes(model) / 1024, "size_csr_kb": csr_bytes(model) / 1024,
        "latency_b1_ms": l1, "latency_b128_ms": l128,
    }


# ------------------------------------------------------------------ steps
def step_baseline(train_d, val_d, test_d):
    CK.mkdir(exist_ok=True)
    seed_all(CFG["seed"])
    model = baseline()
    path = CK / "baseline.pt"
    log_rows = []
    if path.exists():
        ck = torch.load(path)
        model.load_state_dict(ck["state"])
        secs = ck["seconds"]
        print("baseline loaded from checkpoint")
    else:
        t0 = time.perf_counter()
        secs = train(model, train_d, CFG["train"]["baseline_epochs"], CFG["train"]["baseline_lr"], CFG["seed"],
                     log=lambda ep, l: print(f"baseline epoch {ep} loss {l:.4f} val {accuracy(model, *val_d):.4f}", flush=True))
        torch.save({"state": model.state_dict(), "seconds": secs}, path)
    d = describe(model, test_d)
    d["train_seconds"] = secs
    d["val_acc"] = accuracy(model, *val_d)
    write_csv("baseline.csv", list(d.keys()), [list(d.values())])
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})
    return model, d


def step_unstructured(base, train_d, test_d):
    rows, pruned_models = [], {}
    acc0 = accuracy(base, *test_d)
    for mode, fn in (("global", prune.global_masks), ("layer-wise", prune.layer_masks)):
        for s in CFG["prune"]["sparsities"]:
            m = copy.deepcopy(base)
            masks = fn(m, s)
            prune.apply_masks(m, masks)
            before = accuracy(m, *test_d)
            secs = train(m, train_d, CFG["train"]["finetune_epochs"], CFG["train"]["finetune_lr"], CFG["seed"], masks=masks)
            d = describe(m, test_d)
            assert abs(d["weight_sparsity"] - s) < 0.002, "mask not preserved during fine-tuning"
            per_layer = {n: float((w == 0).float().mean()) for n, w in prune.prunable(m).items()}
            rows.append([mode, s, round(before, 4), round(d["test_acc"], 4), round(d["weight_sparsity"], 4),
                         d["weights_nonzero"], round(d["size_dense_kb"], 1), round(d["size_csr_kb"], 1),
                         round(d["latency_b1_ms"], 4), round(d["latency_b128_ms"], 3), d["macs_nonzero"],
                         round(secs, 1), json.dumps({k: round(v, 3) for k, v in per_layer.items()})])
            pruned_models[(mode, s)] = (m, d, secs)
            print(f"{mode:10s} sparsity {s:.2f}: acc before FT {before:.4f} after FT {d['test_acc']:.4f}", flush=True)
    write_csv("unstructured_pruning.csv",
              ["mode", "target_sparsity", "acc_after_pruning_before_finetune", "acc_after_finetune", "actual_sparsity",
               "nonzero_weights", "size_dense_kb", "size_csr_kb", "latency_b1_ms(1 thread)", "latency_b128_ms",
               "macs_nonzero", "finetune_seconds", "per_layer_sparsity"], rows)
    return pruned_models


def step_structured(base, train_d, test_d):
    rows, models = [], {}
    base_params = sum(p.numel() for p in base.parameters())
    for p in CFG["prune"]["filter_fractions"]:
        m = prune.structured_prune(base, p)
        before = accuracy(m, *test_d)
        secs = train(m, train_d, CFG["train"]["finetune_epochs"], CFG["train"]["finetune_lr"], CFG["seed"])
        d = describe(m, test_d)
        removed = 1 - d["params"] / base_params
        rows.append([p, str(m.channels), d["params"], round(removed, 4), round(before, 4), round(d["test_acc"], 4),
                     round(d["size_dense_kb"], 1), round(d["latency_b1_ms"], 4), round(d["latency_b128_ms"], 3),
                     d["macs_dense"], round(secs, 1)])
        models[p] = (m, d, secs, removed)
        print(f"structured {p:.2f}: channels {m.channels} params {d['params']} acc before {before:.4f} after {d['test_acc']:.4f}", flush=True)
    write_csv("structured_pruning.csv",
              ["filters_removed", "channels_left", "params", "fraction_of_params_removed", "acc_before_finetune",
               "acc_after_finetune", "size_dense_kb", "latency_b1_ms(1 thread)", "latency_b128_ms", "macs", "finetune_seconds"], rows)
    return models


def step_sparse_kernel(pruned):
    """Does a sparse kernel help where the dense CPU kernels cannot? fc1 of the globally pruned models:
    dense matrix-vector product against a CSR matrix-vector product (batch 1, 1 thread)."""
    rows = []
    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    x = torch.randn(3136, 1)
    for s in (0.5, 0.75, 0.9, 0.95):
        m = pruned[("global", s)][0]
        w = m.fc1.weight.detach()
        layer_sparsity = float((w == 0).float().mean())
        try:
            ws = w.to_sparse_csr()
            def timeit(fn, n=3000):
                for _ in range(200): fn()
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e6)
                return statistics.median(ts)
            td, ts_ = timeit(lambda: w @ x), timeit(lambda: ws @ x)
            rows.append([s, round(layer_sparsity, 4), round(td, 1), round(ts_, 1), round(td / ts_, 2)])
        except Exception as e:                                   # pragma: no cover
            rows.append([s, round(layer_sparsity, 4), "-", "-", f"unsupported: {type(e).__name__}"])
    torch.set_num_threads(prev)
    write_csv("sparse_kernel_fc1.csv", ["global_target_sparsity", "fc1_sparsity", "dense_matvec_us",
                                        "csr_matvec_us", "speedup_of_csr_over_dense"], rows)
    print("sparse kernel:", rows)


def step_distill(teacher, train_d, test_d):
    d = CFG["distill"]
    T, alpha = d["temperature"], d["alpha"]
    teacher.eval()
    t0 = time.perf_counter()
    with torch.no_grad():
        tl = torch.cat([teacher(train_d[0][i:i + 2000]) for i in range(0, len(train_d[0]), 2000)])
    teacher_secs = time.perf_counter() - t0
    rows, first = [], {}
    for seed in d["seeds"]:
        for name, tlog in (("student, labels only", None), ("student + distillation", tl)):
            seed_all(seed)
            m = student()
            secs = train(m, train_d, CFG["train"]["student_epochs"], CFG["train"]["student_lr"], seed,
                         teacher_logits=tlog, T=T, alpha=alpha)
            if tlog is not None:
                secs += teacher_secs
            acc = accuracy(m, *test_d)
            rows.append([name, seed, round(acc, 4), round(secs, 1)])
            if seed == d["seeds"][0]:
                first[name] = (m, secs, acc)
            print(f"{name:26s} seed {seed}: test acc {acc:.4f}", flush=True)
    write_csv("distillation_runs.csv", ["model", "seed", "test_acc", "train_seconds"], rows)
    summary = []
    for name in ("student, labels only", "student + distillation"):
        accs = [r[2] for r in rows if r[0] == name]
        m = first[name][0]
        l1, l128 = measure_latency(m)
        summary.append([name, sum(p.numel() for p in m.parameters()), round(statistics.mean(accs), 4),
                        round(statistics.pstdev(accs), 4), round(min(accs), 4), round(max(accs), 4),
                        round(dense_bytes(m) / 1024, 1), round(l1, 4), round(l128, 3), m.macs(False)])
    write_csv("distillation.csv", ["model", "params", "test_acc_mean_3_seeds", "std", "min", "max", "size_dense_kb",
                                   "latency_b1_ms(1 thread)", "latency_b128_ms", "macs"], summary)
    return first


def step_summary(base, base_d, pruned, structured, students):
    power = CFG["energy"]["assumed_cpu_power_w"]
    n_train = 55000
    rows = []

    def row(name, d, train_epochs, train_seconds):
        e_inf_uj = d["latency_b128_ms"] / 128 * 1e-3 * power * 1e6          # joules -> microjoules, per image
        rows.append([name, n_train * train_epochs, d["params"], d["weights_nonzero"] if "weights_nonzero" in d else d["params"],
                     round(d["size_dense_kb"], 1), round(d.get("size_csr_kb", d["size_dense_kb"]), 1),
                     d["macs_dense"], d["macs_nonzero"], round(d["latency_b1_ms"], 4), round(d["test_acc"], 4),
                     round(e_inf_uj, 1), round(train_seconds * power, 0)])
    row("baseline (oversized)", base_d, CFG["train"]["baseline_epochs"], base_d["train_seconds"])
    for mode in ("global", "layer-wise"):
        m, d, secs = pruned[(mode, 0.9)]
        row(f"{mode} pruning 90% + FT", d, CFG["train"]["baseline_epochs"] + CFG["train"]["finetune_epochs"],
            base_d["train_seconds"] + secs)
    m, d, secs, _ = structured[0.5]
    d = dict(d); d["macs_nonzero"] = d["macs_dense"]
    row("structured pruning 50% + FT", d, CFG["train"]["baseline_epochs"] + CFG["train"]["finetune_epochs"],
        base_d["train_seconds"] + secs)
    for name in ("student, labels only", "student + distillation"):
        m, secs, acc = students[name]
        d = describe(m, students["_test"])
        if name.endswith("distillation"):
            secs_total = secs + base_d["train_seconds"]          # the student needs the trained teacher
            row(name + " (incl. teacher training)", d, CFG["train"]["student_epochs"] + CFG["train"]["baseline_epochs"], secs_total)
        else:
            row(name, d, CFG["train"]["student_epochs"], secs)
    write_csv("efficiency_pillars.csv",
              ["model", "DATA: training samples seen", "MODEL: parameters", "MODEL: non-zero weights",
               "MODEL: size dense (KB)", "MODEL: size sparse CSR (KB)", "COMPUTE: MACs/img dense",
               "COMPUTE: MACs/img non-zero", "COMPUTE: latency batch 1 (ms)", "accuracy (test)",
               f"ENERGY: inference (uJ/img, est. at {power} W)", f"ENERGY: training (J, est. at {power} W)"], rows)


def plots(base_d, pruned, structured):
    sp_list = CFG["prune"]["sparsities"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for mode, c in (("global", "C0"), ("layer-wise", "C1")):
        ax[0].plot([s * 100 for s in sp_list], [pruned[(mode, s)][1]["test_acc"] for s in sp_list], "o-", color=c, label=mode + " (unstructured)")
    ps = sorted(structured)
    ax[0].plot([structured[p][3] * 100 for p in ps], [structured[p][1]["test_acc"] for p in ps], "s-", color="C3",
               label="structured (filters)")
    ax[0].axhline(base_d["test_acc"], color="gray", ls=":", label="baseline")
    ax[0].set_xlabel("parameters removed (%)"); ax[0].set_ylabel("test accuracy after 3 epochs of fine-tuning")
    ax[0].legend(fontsize=8); ax[0].set_title("Accuracy vs sparsity")
    ax[0].set_ylim(0.8, 0.95)
    for mode, c in (("global", "C0"), ("layer-wise", "C1")):
        ax[1].plot([s * 100 for s in sp_list], [accuracy_cache[(mode, s)] for s in sp_list], "o-", color=c, label=mode)
    ax[1].set_xlabel("weights removed (%)"); ax[1].set_ylabel("test accuracy right after pruning (no fine-tuning)")
    ax[1].set_title("Before fine-tuning"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "sparsity_curves.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode, c in (("global", "C0"), ("layer-wise", "C1")):
        ax[0].plot([s * 100 for s in sp_list], [pruned[(mode, s)][1]["latency_b1_ms"] for s in sp_list], "o-", color=c, label=mode + " (masked dense)")
    ax[0].plot([structured[p][3] * 100 for p in ps], [structured[p][1]["latency_b1_ms"] for p in ps], "s-", color="C3", label="structured")
    ax[0].set_xlabel("parameters removed (%)"); ax[0].set_ylabel("latency, batch 1, 1 thread (ms)"); ax[0].legend(fontsize=8)
    ax[0].set_title("Latency")
    ax[1].plot([s * 100 for s in sp_list], [pruned[("global", s)][1]["size_dense_kb"] for s in sp_list], "o-", label="dense file")
    ax[1].plot([s * 100 for s in sp_list], [pruned[("global", s)][1]["size_csr_kb"] for s in sp_list], "o-", label="CSR sparse file")
    ax[1].set_xlabel("weights removed (%)"); ax[1].set_ylabel("size on disk (KB)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Dense vs sparse format (global pruning)")
    fig.tight_layout(); fig.savefig(RES / "latency_size.png", dpi=130)


accuracy_cache = {}

if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    seed_all(CFG["seed"])
    train_d, val_d, test_d = load_data()
    (RES / "versions.txt").write_text(
        f"python {platform.python_version()}\ntorch {torch.__version__}\nnumpy {np.__version__}\n"
        f"scipy {__import__('scipy').__version__}\nthreads {torch.get_num_threads()}\ncpu_count {os.cpu_count()}\n"
        f"platform {platform.platform()}\n")
    base, base_d = step_baseline(train_d, val_d, test_d)
    # accuracy right after pruning (needed by the plot) is stored in the csv; reload it for plotting
    pruned = step_unstructured(base, train_d, test_d)
    for r in csv.DictReader(open(RES / "unstructured_pruning.csv")):
        accuracy_cache[(r["mode"], float(r["target_sparsity"]))] = float(r["acc_after_pruning_before_finetune"])
    structured = step_structured(base, train_d, test_d)
    step_sparse_kernel(pruned)
    students = step_distill(base, train_d, test_d)
    students["_test"] = test_d
    step_summary(base, base_d, pruned, structured, students)
    plots(base_d, pruned, structured)
    print("done")
