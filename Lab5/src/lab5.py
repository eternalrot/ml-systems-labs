"""Lab 5 driver. Run from Lab5/:  python src/lab5.py"""
import csv
import os
import platform
import random
import statistics
import time

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from engine import Value, zero_grad

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
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


# ---------------------------------------------------------------- part 4
def check_gradients():
    """f(a,b,c) = sum(relu((a*b + c)**2)) on vectors, three ways."""
    seed_all()
    rng = np.random.default_rng(SEED)
    n = 8
    a0, b0, c0 = (rng.normal(size=n) for _ in range(3))

    def f_engine(a, b, c):
        return sum((((x * y) + z) ** 2).relu() for x, y, z in zip(a, b, c))

    # (1) own engine
    a = [Value(v) for v in a0]
    b = [Value(v) for v in b0]
    c = [Value(v) for v in c0]
    out = f_engine(a, b, c)
    out.backward()
    g_own = np.array([[v.grad for v in vs] for vs in (a, b, c)])

    # (2) PyTorch autograd (float64)
    ta, tb, tc = (torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (a0, b0, c0))
    tout = ((ta * tb + tc) ** 2).relu().sum()
    tout.backward()
    g_torch = np.array([t.grad.numpy() for t in (ta, tb, tc)])

    # (3) numerical central differences, h = 1e-6
    def f_num(av, bv, cv):
        return float(np.sum(np.maximum((av * bv + cv) ** 2, 0.0)))
    h = 1e-6
    g_num = np.zeros_like(g_own)
    base = [a0.copy(), b0.copy(), c0.copy()]
    for k in range(3):
        for i in range(n):
            up = [x.copy() for x in base]
            dn = [x.copy() for x in base]
            up[k][i] += h
            dn[k][i] -= h
            g_num[k, i] = (f_num(*up) - f_num(*dn)) / (2 * h)

    rows = [
        ["forward value: own engine", out.data],
        ["forward value: torch", float(tout)],
        ["max|own - torch autograd| (grad)", float(np.abs(g_own - g_torch).max())],
        ["max|own - numerical| (grad)", float(np.abs(g_own - g_num).max())],
        ["max|torch - numerical| (grad)", float(np.abs(g_torch - g_num).max())],
        ["forward diff |own - torch|", abs(out.data - float(tout))],
    ]
    write_csv("gradient_check.csv", ["quantity", "value"], rows)

    # accumulation test: x is used twice, so d/dx (x*x + x) = 2x + 1 (an overwrite bug would give x or 1)
    x = Value(3.0)
    y = x * x + x
    y.backward()
    # a node reused along two paths (diamond): z = (x+1)*(x+2)  -> dz/dx = 2x + 3
    x2 = Value(3.0)
    z = (x2 + 1) * (x2 + 2)
    z.backward()
    write_csv("accumulation_check.csv", ["case", "expected", "got"],
              [["d/dx (x*x + x) at x=3", 7.0, x.grad], ["d/dx (x+1)(x+2) at x=3", 9.0, x2.grad]])
    return rows


# ---------------------------------------------------------------- part 5
H = 8
EPOCHS = 100
LR = 0.5


def make_data():
    X, y = make_moons(n_samples=300, noise=0.2, random_state=SEED)
    return train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)


def init_params():
    rng = np.random.default_rng(SEED)
    W1 = rng.normal(0, 1.0 / np.sqrt(2), size=(H, 2))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 1.0 / np.sqrt(H), size=(1, H))
    b2 = np.zeros(1)
    return W1, b1, W2, b2


def train_engine(Xtr, ytr, Xte, yte):
    W1, b1, W2, b2 = init_params()
    w1 = [[Value(v) for v in row] for row in W1]
    bb1 = [Value(v) for v in b1]
    w2 = [Value(v) for v in W2[0]]
    bb2 = Value(b2[0])
    params = [p for row in w1 for p in row] + bb1 + w2 + [bb2]

    def forward(x):
        hid = [(sum(w * xi for w, xi in zip(row, x)) + bj).relu() for row, bj in zip(w1, bb1)]
        return (sum(w * h for w, h in zip(w2, hid)) + bb2).sigmoid()

    losses = []
    n = len(Xtr)
    for _ in range(EPOCHS):
        for p in params:
            p.grad = 0.0
        loss = Value(0.0)
        for x, t in zip(Xtr, ytr):
            p = forward(x)
            # BCE = -[t log p + (1-t) log(1-p)]
            loss = loss + (-(t * p.log() + (1 - t) * (1 - p).log()))
        loss = loss * (1.0 / n)
        loss.backward()
        for p in params:
            p.data -= LR * p.grad
        losses.append(loss.data)
    acc = float(np.mean([(forward(x).data > 0.5) == bool(t) for x, t in zip(Xte, yte)]))
    return losses, acc


def train_torch(Xtr, ytr, Xte, yte):
    W1, b1, W2, b2 = init_params()
    dt = torch.float64
    w1 = torch.tensor(W1, dtype=dt, requires_grad=True)
    bb1 = torch.tensor(b1, dtype=dt, requires_grad=True)
    w2 = torch.tensor(W2, dtype=dt, requires_grad=True)
    bb2 = torch.tensor(b2, dtype=dt, requires_grad=True)
    X = torch.tensor(Xtr, dtype=dt)
    y = torch.tensor(ytr, dtype=dt)
    params = [w1, bb1, w2, bb2]

    def forward(X):
        return torch.sigmoid(torch.relu(X @ w1.T + bb1) @ w2.T + bb2).squeeze(1)

    losses = []
    for _ in range(EPOCHS):
        for p in params:
            p.grad = None
        p_ = forward(X)
        loss = torch.nn.functional.binary_cross_entropy(p_, y)
        loss.backward()
        with torch.no_grad():
            for p in params:
                p -= LR * p.grad
        losses.append(loss.item())
    with torch.no_grad():
        acc = float(((forward(torch.tensor(Xte, dtype=dt)) > 0.5).numpy() == yte).mean())
    return losses, acc


def part5():
    Xtr, Xte, ytr, yte = make_data()
    # warm-up (also gives the curves)
    l_own, acc_own = train_engine(Xtr, ytr, Xte, yte)
    l_th, acc_th = train_torch(Xtr, ytr, Xte, yte)
    write_csv("training_curves.csv", ["epoch", "loss_own_engine", "loss_torch"],
              [[i + 1, a, b] for i, (a, b) in enumerate(zip(l_own, l_th))])
    t_own, t_th = [], []
    for _ in range(5):
        t0 = time.perf_counter(); train_engine(Xtr, ytr, Xte, yte); t_own.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); train_torch(Xtr, ytr, Xte, yte); t_th.append(time.perf_counter() - t0)
    rows = [
        ["own engine", f"{statistics.median(t_own):.4f}", f"{min(t_own):.4f}", f"{max(t_own):.4f}",
         f"{l_own[0]:.4f}", f"{l_own[-1]:.4f}", f"{acc_own:.4f}"],
        ["pytorch (float64)", f"{statistics.median(t_th):.4f}", f"{min(t_th):.4f}", f"{max(t_th):.4f}",
         f"{l_th[0]:.4f}", f"{l_th[-1]:.4f}", f"{acc_th:.4f}"],
    ]
    write_csv("moons_speed.csv", ["implementation", "median_s_per_100_epochs", "min_s", "max_s",
                                  "first_epoch_loss", "last_epoch_loss", "test_accuracy"], rows)
    write_csv("moons_speed_ratio.csv", ["quantity", "value"],
              [["median time ratio own/torch", statistics.median(t_own) / statistics.median(t_th)],
               ["max |loss_own - loss_torch| over epochs", max(abs(a - b) for a, b in zip(l_own, l_th))],
               ["train rows", len(Xtr)], ["test rows", len(Xte)], ["epochs", EPOCHS], ["lr", LR]])
    return rows


# ---------------------------------------------------------------- part 6
def median_time(fn, repeats=5, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        ts.append((time.perf_counter() - t0) / iters)
    return statistics.median(ts)


def part6():
    seed_all()
    model = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 512), torch.nn.ReLU(),
                                torch.nn.Linear(512, 10))
    model.eval()
    x = torch.randn(64, 256)

    # ---- graph inspection ----
    lines = []
    xin = x.clone().requires_grad_(False)
    out = model(xin)
    loss = out.logsumexp(dim=1).mean()
    lines.append("== autograd graph (grad_fn chain, walked from the loss) ==")
    seen = set()

    def walk(fn, depth=0):
        if fn is None or id(fn) in seen:
            return
        seen.add(id(fn))
        lines.append("  " * depth + type(fn).__name__)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)
    walk(loss.grad_fn)
    lines.append("")
    lines.append("== torch.fx symbolic trace ==")
    gm = torch.fx.symbolic_trace(model)
    lines.append(str(gm.graph))
    lines.append("")
    lines.append("== tensors: shape / dtype / stride / contiguity ==")

    def desc(name, t):
        lines.append(f"{name:28s} shape={tuple(t.shape)} dtype={t.dtype} stride={t.stride()} "
                     f"contiguous={t.is_contiguous()}")
    desc("input x", x)
    desc("Linear0.weight", model[0].weight)
    desc("Linear0.weight.T (view)", model[0].weight.T)
    desc("x[:, ::2] (strided view)", x[:, ::2])
    desc("x.contiguous()", x.contiguous())
    desc("output", out)
    open(os.path.join(RES, "graph_inspection.txt"), "w").write("\n".join(lines) + "\n")

    # ---- eager vs trace vs compile ----
    rows = []
    with torch.no_grad():
        eager = lambda: model(x)
        t_eager = median_time(eager)
        rows.append(["eager", f"{t_eager * 1e6:.1f}", "1.00", "ok"])

        traced = torch.jit.trace(model, x)
        traced = torch.jit.freeze(traced)
        t_trace = median_time(lambda: traced(x), warmup=30)
        rows.append(["torch.jit.trace + freeze", f"{t_trace * 1e6:.1f}", f"{t_eager / t_trace:.2f}", "ok"])

        t0 = time.perf_counter()
        try:
            compiled = torch.compile(model)
            compiled(x)                      # first call = compilation
            first = time.perf_counter() - t0
            t_comp = median_time(lambda: compiled(x), warmup=20)
            rows.append(["torch.compile (first call, s)", f"{first:.2f}", "-", "ok"])
            rows.append(["torch.compile (after warm-up)", f"{t_comp * 1e6:.1f}", f"{t_eager / t_comp:.2f}", "ok"])
        except Exception as e:                # e.g. no C++ compiler on Windows
            rows.append(["torch.compile", "-", "-", "UNSUPPORTED: " + type(e).__name__ + ": " + str(e).splitlines()[0][:120]])
        # the same models on a tiny input: where fixed overhead dominates
        xs = torch.randn(1, 256)
        t_e1 = median_time(lambda: model(xs), iters=200)
        t_t1 = median_time(lambda: traced(xs), iters=200, warmup=30)
        rows.append(["eager, batch=1", f"{t_e1 * 1e6:.1f}", "1.00", "ok"])
        rows.append(["trace+freeze, batch=1", f"{t_t1 * 1e6:.1f}", f"{t_e1 / t_t1:.2f}", "ok"])
    write_csv("eager_vs_compiled.csv", ["mode", "value (us per forward; compile first-call in s)", "speedup_vs_eager", "status"], rows)

    # ---- non-contiguous view ----
    big = torch.randn(2048, 2048)
    cont = big
    nonc = big.t()                          # a view: same memory, strides swapped
    assert not nonc.is_contiguous()
    ops = {
        "relu": lambda t: torch.relu(t),
        "add (t + t)": lambda t: t + t,
        "sum(dim=1)": lambda t: t.sum(dim=1),
    }
    crow = []
    with torch.no_grad():
        for name, op in ops.items():
            tc = median_time(lambda: op(cont), iters=20, warmup=5)
            tn = median_time(lambda: op(nonc), iters=20, warmup=5)
            crow.append([name, f"{tc * 1e3:.3f}", f"{tn * 1e3:.3f}", f"{tn / tc:.2f}"])
        t_copy = median_time(lambda: nonc.contiguous(), iters=20, warmup=5)
        crow.append(["nonc.contiguous() (copy cost)", "-", f"{t_copy * 1e3:.3f}", "-"])
        mm_c = median_time(lambda: cont @ cont, iters=5, warmup=2)
        mm_n = median_time(lambda: cont @ nonc, iters=5, warmup=2)
        crow.append(["matmul A @ B vs A @ B.T", f"{mm_c * 1e3:.3f}", f"{mm_n * 1e3:.3f}", f"{mm_n / mm_c:.2f}"])
    write_csv("noncontiguous.csv", ["operation", "contiguous_ms", "non_contiguous_ms", "ratio"], crow)
    return rows, crow


if __name__ == "__main__":
    seed_all()
    with open(os.path.join(RES, "versions.txt"), "w") as f:
        f.write(f"python {platform.python_version()}\ntorch {torch.__version__}\nnumpy {np.__version__}\n"
                f"threads {torch.get_num_threads()}\nplatform {platform.platform()}\n")
    for r in check_gradients():
        print(r)
    print(part5())
    r6 = part6()
    print(r6[0]); print(r6[1])
