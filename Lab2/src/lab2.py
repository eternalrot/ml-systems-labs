"""Lab 2 driver. Run from Lab2/: python src/lab2.py"""
import csv
import os
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import load_data
from nn import (SEED, accuracy, backward, cross_entropy, forward, init_params,
                loss_only, set_seeds)

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RES, exist_ok=True)
set_seeds()
X_tr, X_te, Y_tr, Y_te, y_tr, y_te = load_data()
print("shapes", X_tr.shape, X_te.shape, Y_tr.shape)


# ---- Step 3/4: forward sanity checks -----------------------------------
def sanity(width):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    P, _ = forward(p, X_tr[:32])
    row_err = float(np.max(np.abs(P.sum(axis=1) - 1.0)))
    loss = float(loss_only(p, X_tr, Y_tr))
    return P.shape, row_err, loss


with open(os.path.join(RES, "sanity_checks.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width", "output_shape", "max_row_sum_error", "untrained_loss", "ln10", "row_sum_ok"])
    for W in (16, 64):
        shape, err, loss = sanity(W)
        w.writerow([W, shape, f"{err:.2e}", f"{loss:.4f}", f"{np.log(10):.4f}", err < 1e-6])
        print(f"W={W} output {shape} max|rowsum-1|={err:.2e} untrained loss={loss:.4f} (ln10={np.log(10):.4f})")


# ---- Step 5: numerical gradient check -----------------------------------
def grad_check(p, X, Y, n_checks=10, eps=1e-5, seed=SEED):
    """Nudge n random weights (non-zero analytic gradient) by +-eps, compare slopes."""
    rng = np.random.default_rng(seed)
    P, cache = forward(p, X)
    g = backward(p, cache, P, Y)
    # 3 weights from W1, 3 from W2, 4 from W3 (only those with non-zero analytic gradient)
    chosen = []
    for k, cnt in (("W1", 3), ("W2", 3), ("W3", 4)):
        cand_k = [(k, idx) for idx in zip(*np.nonzero(np.abs(g[k]) > 1e-8))]
        chosen += [cand_k[j] for j in rng.choice(len(cand_k), cnt, replace=False)]
    cand, pick = chosen, range(len(chosen))
    rows, ok = [], 0
    for j in pick:
        name, idx = cand[j]
        idx = tuple(int(i) for i in idx)
        old = p[name][idx]
        p[name][idx] = old + eps
        lp = loss_only(p, X, Y)
        p[name][idx] = old - eps
        lm = loss_only(p, X, Y)
        p[name][idx] = old
        num, ana = (lp - lm) / (2 * eps), g[name][idx]
        rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
        ok += bool(rel < 1e-6)
        rows.append([name, str(idx), f"{ana:.6e}", f"{num:.6e}", f"{rel:.2e}", bool(rel < 1e-6)])
    return rows, ok


def trained_params(width, lr=0.1, epochs=60, bs=32):
    """Network after some training, where gradients are not vanishingly small."""
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    for _ in range(epochs):
        perm = rng.permutation(len(X_tr))
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            P, c = forward(p, X_tr[b])
            g = backward(p, c, P, Y_tr[b])
            for k in p:
                p[k] -= lr * g[k]
    return p


with open(os.path.join(RES, "gradient_check.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["point", "width", "param", "index", "analytic", "numeric", "relative_error", "pass(<1e-6)"])
    for W in (16, 64):
        pt = {"trained_60_epochs": trained_params(W),
              "init_N(0,0.01)": init_params(W, np.random.default_rng(SEED))}
        for name, p in pt.items():
            rows, ok = grad_check(p, X_tr[:32], Y_tr[:32])
            w.writerows([[name, W] + r for r in rows])
            print(f"gradient check W={W} at {name}: {ok}/10 weights with rel.error < 1e-6")

# ---- Step 6: training ----------------------------------------------------
def train(width, lr, epochs=50, bs=32, he=False):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    if he:  # EXTRA experiment (not part of the manual): He initialisation
        for k in ("W1", "W2", "W3"):
            p[k] = rng.normal(0.0, np.sqrt(2.0 / p[k].shape[0]), p[k].shape)
    n = X_tr.shape[0]
    hist = []
    for ep in range(1, epochs + 1):
        perm = rng.permutation(n)
        losses = []
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            P, cache = forward(p, X_tr[b])
            losses.append(cross_entropy(P, Y_tr[b]))
            g = backward(p, cache, P, Y_tr[b])
            for k in p:
                p[k] -= lr * g[k]
        tr_loss = float(np.mean(losses))
        te_acc = accuracy(p, X_te, y_te)
        hist.append((ep, tr_loss, te_acc))
        if ep % 10 == 0 or ep == 1:
            print(f"  {'He ' if he else ''}W={width} lr={lr} epoch {ep:2d} loss {tr_loss:.4f} test acc {te_acc:.4f}")
    return hist


LRS, WIDTHS = [0.001, 0.01, 0.1], [16, 64]
final, curves = [], {}
with open(os.path.join(RES, "training_log.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width", "lr", "epoch", "train_loss", "test_accuracy"])
    for W in WIDTHS:
        for lr in LRS:
            h = train(W, lr)
            curves[(W, lr)] = h
            w.writerows([[W, lr, e, f"{l:.5f}", f"{a:.4f}"] for e, l, a in h])
            final.append([W, lr, round(h[-1][1], 4), round(h[-1][2], 4)])

with open(os.path.join(RES, "final_accuracy.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width", "lr", "final_train_loss", "final_test_accuracy"])
    w.writerows(final)

def plot_curves(curves, path, title_suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, W in zip(axes, WIDTHS):
        for lr in LRS:
            h = curves[(W, lr)]
            ax.plot([e for e, _, _ in h], [l for _, l, _ in h], label=f"lr={lr}")
        ax.set_title(f"Hidden width W={W}{title_suffix}")
        ax.set_xlabel("epoch")
        ax.set_yscale("log")
        ax.grid(alpha=.3)
        ax.legend()
    axes[0].set_ylabel("training loss (cross-entropy, log scale)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


plot_curves(curves, os.path.join(RES, "loss_curves.png"))

# EXTRA (not required by the manual): same grid with He initialisation
he_curves, he_final = {}, []
for W in WIDTHS:
    for lr in LRS:
        h = train(W, lr, he=True)
        he_curves[(W, lr)] = h
        he_final.append([W, lr, round(h[-1][1], 4), round(h[-1][2], 4)])
plot_curves(he_curves, os.path.join(RES, "extra_loss_curves_he_init.png"), " (He init, extra)")
with open(os.path.join(RES, "extra_final_accuracy_he_init.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width", "lr", "final_train_loss", "final_test_accuracy"])
    w.writerows(he_final)


# ---- Step 7: timing ------------------------------------------------------
def time_passes(width, bs, runs=100):
    rng = np.random.default_rng(SEED)
    p = init_params(width, rng)
    X, Y = X_tr[:bs], Y_tr[:bs]
    for _ in range(10):                       # warm-up
        P, c = forward(p, X); backward(p, c, P, Y)
    tf, tb, tm = [], [], []
    for _ in range(runs):
        t0 = time.perf_counter(); P, c = forward(p, X); t1 = time.perf_counter()
        backward(p, c, P, Y); t2 = time.perf_counter()
        tf.append((t1 - t0) * 1e6); tb.append((t2 - t1) * 1e6)
    return statistics.median(tf), statistics.median(tb)


with open(os.path.join(RES, "timing.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["width", "batch", "forward_median_us", "backward_median_us", "backward_over_forward"])
    for W in WIDTHS:
        for bs in (32, len(X_tr)):
            tf, tb = time_passes(W, bs)
            w.writerow([W, bs, round(tf, 1), round(tb, 1), round(tb / tf, 2)])
            print(f"W={W} batch={bs}: forward {tf:.1f} us, backward {tb:.1f} us, ratio {tb/tf:.2f}")
