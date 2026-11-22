"""Lab 8 driver: manual quantization, dynamic and static quantization, ONNX export, variant comparison.
Run from Lab8/:   python src/lab8.py          (about 8-12 minutes, trained float models are cached)
"""
import copy
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import common as C                                     # noqa: E402
import prune                                           # noqa: E402
from models import baseline, student                   # noqa: E402
from qmodels import static_quantize                    # noqa: E402

CFG = yaml.safe_load(open(ROOT / "config.yaml"))
RES, CK, MOD = ROOT / "results", ROOT / "checkpoints", ROOT / "models"


def write_csv(name, header, rows):
    with open(RES / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def backend_name():
    b = CFG["quant"]["backend"]
    if b != "auto":
        return b
    return "qnnpack" if platform.machine().lower() in ("arm64", "aarch64") else "fbgemm"


# ------------------------------------------------------------------ float models
def float_models(train_d, val_d, test_d):
    """Baseline (teacher), structured-pruned baseline, distilled student, plain student - cached in checkpoints/."""
    CK.mkdir(exist_ok=True)
    T, A = CFG["distill"]["temperature"], CFG["distill"]["alpha"]
    t = CFG["train"]
    path = CK / "float_models.pt"
    if path.exists():
        ck = torch.load(path, weights_only=False)
        base = baseline(); base.load_state_dict(ck["baseline"])
        pruned = prune.structured_prune(base, CFG["prune"]["filter_fraction"]); pruned.load_state_dict(ck["pruned"])
        dist = student(); dist.load_state_dict(ck["distilled"])
        plain = student(); plain.load_state_dict(ck["plain"])
        print("float models loaded from checkpoint")
        return {"baseline": base, "pruned": pruned, "distilled": dist, "plain": plain}
    C.seed_all(CFG["seed"])
    base = baseline()
    C.train(base, train_d, t["baseline_epochs"], t["baseline_lr"], CFG["seed"], t["batch_size"],
            log=lambda ep, l: print(f"baseline epoch {ep} loss {l:.4f}", flush=True))
    pruned = prune.structured_prune(base, CFG["prune"]["filter_fraction"])
    C.train(pruned, train_d, t["finetune_epochs"], t["finetune_lr"], CFG["seed"], t["batch_size"])
    base.eval()
    with torch.no_grad():
        tl = torch.cat([base(train_d[0][i:i + 1000]) for i in range(0, len(train_d[0]), 1000)])
    C.seed_all(CFG["seed"]); dist = student()
    C.train(dist, train_d, t["student_epochs"], t["student_lr"], CFG["seed"], t["batch_size"], teacher_logits=tl, T=T, alpha=A)
    C.seed_all(CFG["seed"]); plain = student()
    C.train(plain, train_d, t["student_epochs"], t["student_lr"], CFG["seed"], t["batch_size"])
    torch.save({"baseline": base.state_dict(), "pruned": pruned.state_dict(), "distilled": dist.state_dict(),
                "plain": plain.state_dict()}, path)
    return {"baseline": base, "pruned": pruned, "distilled": dist, "plain": plain}


# ------------------------------------------------------------------ 1. manual quantization
def quantize_symmetric(w, bits, per_channel):
    """scale = max|W| / (2^(bits-1) - 1), zero point 0; per-channel = one scale per output channel (row)."""
    qmax = 2 ** (bits - 1) - 1
    if per_channel:
        amax = w.abs().flatten(1).max(dim=1).values.clamp_min(1e-12)
        scale = (amax / qmax).view(-1, *([1] * (w.dim() - 1)))
    else:
        scale = (w.abs().max() / qmax).clamp_min(1e-12)
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q, scale, q * scale


def step_manual(models, test_d):
    base = models["baseline"]
    rows = []
    for lname, layer in (("fc1 (320x3136)", base.fc1), ("conv3 (64x32x3x3)", base.conv3)):
        w = layer.weight.detach()
        for bits in (8, 4, 2):
            for pc in (False, True):
                q, scale, wh = quantize_symmetric(w, bits, pc)
                err = wh - w
                mse = float((err ** 2).mean())
                snr = 10 * math.log10(float((w ** 2).mean()) / mse) if mse > 0 else float("inf")
                rows.append([lname, bits, "per-channel" if pc else "per-tensor", float(scale.flatten()[0]) if not pc else float(scale.mean()),
                             f"{mse:.3e}", f"{float(err.abs().max()):.3e}", round(snr, 2), int(q.unique().numel()),
                             f"{float((q == 0).float().mean()):.3f}"])
    write_csv("manual_quantization.csv", ["tensor", "bits", "scaling", "scale (mean for per-channel)", "MSE",
                                          "max_abs_error", "SNR_dB", "distinct_levels_used", "share_of_zeros"], rows)
    # effect on the whole model: all weights fake-quantized, activations stay float
    acc_rows = [["float32 (original)", "-", round(C.accuracy(base, *test_d), 4)]]
    for bits in (8, 4, 2):
        for pc in (False, True):
            m = copy.deepcopy(base)
            with torch.no_grad():
                for n, layer in m.layers().items():
                    layer.weight.copy_(quantize_symmetric(layer.weight, bits, pc)[2])
            acc_rows.append([f"{bits}-bit", "per-channel" if pc else "per-tensor", round(C.accuracy(m, *test_d), 4)])
    write_csv("manual_quantization_model_accuracy.csv", ["weights", "scaling", "test_acc"], acc_rows)
    # a worked example on 6 numbers for the report
    w = base.fc1.weight.detach().flatten()[:6]
    q, s, wh = quantize_symmetric(w.view(1, -1), 8, False)
    print("example:", w.tolist(), "scale", float(s), "int8", q.int().tolist(), "back", wh.tolist())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for a, lname in zip(ax, ("fc1 (320x3136)", "conv3 (64x32x3x3)")):
        for pc, c in ((False, "C0"), (True, "C1")):
            xs = [r[1] for r in rows if r[0] == lname and r[2] == ("per-channel" if pc else "per-tensor")]
            ys = [float(r[4]) for r in rows if r[0] == lname and r[2] == ("per-channel" if pc else "per-tensor")]
            a.plot(xs, ys, "o-", color=c, label="per-channel" if pc else "per-tensor")
        a.set_yscale("log"); a.set_xlabel("bits"); a.set_ylabel("MSE"); a.set_title(lname); a.invert_xaxis(); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RES / "manual_quantization.png", dpi=130)


# ------------------------------------------------------------------ 2. dynamic, 3. static
def step_dynamic(models, test_d, backend):
    torch.backends.quantized.engine = backend
    rows, out = [], {}
    for name in ("baseline", "pruned", "distilled"):
        f = models[name]
        q = torch.ao.quantization.quantize_dynamic(copy.deepcopy(f), {nn.Linear}, dtype=torch.qint8)
        l1, l128 = C.measure_latency(q, CFG)
        fl1, fl128 = C.measure_latency(f, CFG)
        acc = C.accuracy(q, *test_d)
        rows.append([name, backend, round(C.accuracy(f, *test_d), 4), round(acc, 4), round(C.state_bytes(f) / 1024, 1),
                     round(C.state_bytes(q) / 1024, 1), round(fl1, 4), round(l1, 4), round(fl128, 3), round(l128, 3)])
        out[name] = (q, acc, C.state_bytes(q) / 1024, l1, l128)
        print(f"dynamic {name}: acc {acc:.4f}", flush=True)
    write_csv("dynamic_quantization.csv", ["model", "backend", "acc_float", "acc_dynamic_int8", "size_float_kb",
                                           "size_dynamic_kb", "latency_b1_float_ms", "latency_b1_dynamic_ms",
                                           "latency_b128_float_ms", "latency_b128_dynamic_ms"], rows)
    return out


def calib_images(train_d, n, seed, label=None, noise=False):
    X, y = train_d
    g = torch.Generator().manual_seed(seed)
    if noise:
        return torch.randn(n, 1, 28, 28, generator=g)
    idx = torch.nonzero(y == label).flatten() if label is not None else torch.arange(len(X))
    return X[idx[torch.randperm(len(idx), generator=g)[:n]]]


def step_static(models, train_d, test_d, backend):
    n, seeds = CFG["quant"]["calibration_images"], CFG["quant"]["calibration_seeds"]
    out, rows = {}, []
    for name in ("baseline", "pruned", "distilled"):
        f = models[name]
        q = static_quantize(f, calib_images(train_d, n, CFG["seed"]), backend)
        l1, l128 = C.measure_latency(q, CFG)
        acc = C.accuracy(q, *test_d)
        rows.append([name, backend, n, round(C.accuracy(f, *test_d), 4), round(acc, 4), round(C.state_bytes(f) / 1024, 1),
                     round(C.state_bytes(q) / 1024, 1), round(l1, 4), round(l128, 3)])
        out[name] = (q, acc, C.state_bytes(q) / 1024, l1, l128)
        print(f"static {name}: acc {acc:.4f}", flush=True)
    write_csv("static_quantization.csv", ["model", "backend", "calibration_images", "acc_float", "acc_static_int8",
                                          "size_float_kb", "size_static_kb", "latency_b1_ms", "latency_b128_ms"], rows)
    # why the calibration set must be representative: baseline calibrated on different kinds of data
    cal_rows = []
    f = models["baseline"]
    def run(desc, imgs):
        q = static_quantize(f, imgs, backend)
        cal_rows.append([desc, len(imgs), round(C.accuracy(q, *test_d), 4)])
    for s in seeds:
        run(f"random training images (seed {s})", calib_images(train_d, n, s))
    run("random training images, only 50", calib_images(train_d, 50, CFG["seed"]))
    run("random training images, only 10", calib_images(train_d, 10, CFG["seed"]))
    run("only class 0 (T-shirt)", calib_images(train_d, n, CFG["seed"], label=0))
    run("only class 9 (ankle boot)", calib_images(train_d, n, CFG["seed"], label=9))
    run("random noise instead of images", calib_images(train_d, n, CFG["seed"], noise=True))
    run("real images scaled by 0.1 (range 10x too narrow)", calib_images(train_d, n, CFG["seed"]) * 0.1)
    run("blank black images (all pixels the same)", torch.full((n, 1, 28, 28), float(train_d[0].min())))
    write_csv("calibration_study.csv", ["calibration data", "images", "test_acc_static_int8 (baseline)"], cal_rows)
    for r in cal_rows:
        print(r)
    return out


# ------------------------------------------------------------------ 4. ONNX
def step_onnx(models, test_d):
    MOD.mkdir(exist_ok=True)
    rows, out = [], {}
    Xc = test_d[0][:CFG["onnx"]["check_images"]]
    for name in ("baseline", "pruned", "distilled"):
        m = models[name].eval()
        path = MOD / f"{name}.onnx"
        torch.onnx.export(m, torch.randn(1, 1, 28, 28), str(path), input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=CFG["onnx"]["opset"])
        onnx.checker.check_model(onnx.load(str(path)))
        so1 = ort.SessionOptions(); so1.intra_op_num_threads = 1; so1.inter_op_num_threads = 1
        s1 = ort.InferenceSession(str(path), so1, providers=["CPUExecutionProvider"])
        sN = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            ref = m(Xc).numpy()
        got = s1.run(None, {"image": Xc.numpy()})[0]
        diff = float(np.abs(ref - got).max())
        same = float((ref.argmax(1) == got.argmax(1)).mean())
        assert diff < CFG["onnx"]["tolerance"], f"ONNX output differs by {diff}"
        f1 = lambda x: s1.run(None, {"image": x.numpy()})
        fN = lambda x: sN.run(None, {"image": x.numpy()})
        l1 = C.latency_ms(f1, C.img_batch, 1, CFG["latency"]["batch1_runs"], CFG["latency"]["warmup"])
        l128 = C.latency_ms(fN, C.img_batch, 128, CFG["latency"]["batch128_runs"], CFG["latency"]["warmup"])
        tl1, tl128 = C.measure_latency(m, CFG)
        acc = float((sN.run(None, {"image": test_d[0].numpy()})[0].argmax(1) == test_d[1].numpy()).mean())
        rows.append([name, CFG["onnx"]["opset"], f"{diff:.2e}", CFG["onnx"]["tolerance"], same, round(acc, 4),
                     round(path.stat().st_size / 1024, 1), round(tl1, 4), round(l1, 4), round(tl128, 3), round(l128, 3), ort.__version__])
        out[name] = (acc, path.stat().st_size / 1024, l1, l128)
        print(f"onnx {name}: max diff {diff:.2e}, acc {acc:.4f}, ort b1 {l1:.3f} ms vs torch {tl1:.3f} ms", flush=True)
    write_csv("onnx_validation.csv", ["model", "opset", "max_abs_diff_vs_torch", "tolerance", "same_argmax_share(1000 imgs)",
                                      "test_acc_onnxruntime", "onnx_file_kb", "latency_b1_torch_ms", "latency_b1_onnxruntime_ms",
                                      "latency_b128_torch_ms", "latency_b128_onnxruntime_ms", "onnxruntime_version"], rows)
    return out


# ------------------------------------------------------------------ 5. variants
def step_variants(models, dyn, sta, onx, test_d, backend):
    V = []                                                     # (name, runtime, acc, size_kb, lat1, lat128, params)
    def add(name, runtime, acc, size, l1, l128, params):
        V.append([name, runtime, round(acc, 4), round(size, 1), round(l1, 4), round(l128, 3), params])
    for name, label in (("baseline", "baseline fp32"), ("pruned", "pruned (structured 50%) fp32"),
                        ("distilled", "distilled student fp32")):
        m = models[name]
        l1, l128 = C.measure_latency(m, CFG)
        add(label, "PyTorch", C.accuracy(m, *test_d), C.state_bytes(m) / 1024, l1, l128, sum(p.numel() for p in m.parameters()))
    for i, (name, label) in enumerate((("baseline", "baseline fp32"), ("pruned", "pruned (structured 50%) fp32"),
                                       ("distilled", "distilled student fp32"))):
        add(label, "ONNX Runtime", onx[name][0], onx[name][1], onx[name][2], onx[name][3], V[i][6])
    q = dyn["baseline"]; add("baseline dynamic int8", f"PyTorch ({backend})", q[1], q[2], q[3], q[4], V[0][6])
    q = sta["baseline"]; add("baseline static int8", f"PyTorch ({backend})", q[1], q[2], q[3], q[4], V[0][6])
    q = sta["pruned"]; add("pruned + static int8", f"PyTorch ({backend})", q[1], q[2], q[3], q[4], V[1][6])
    q = sta["distilled"]; add("distilled + static int8", f"PyTorch ({backend})", q[1], q[2], q[3], q[4], V[2][6])
    write_csv("variants.csv", ["variant", "runtime", "test_acc", "size_kb", "latency_b1_ms(1 thread)", "latency_b128_ms", "params"], V)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for r in V:
        ax[0].scatter(r[4], r[2], s=45); ax[0].annotate(r[0].replace(" fp32", "") + ("" if r[1] == "PyTorch" or "(" in r[1] else " [ORT]"), (r[4], r[2]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].scatter(r[3], r[2], s=45); ax[1].annotate(r[0].replace(" fp32", "") + ("" if r[1] == "PyTorch" or "(" in r[1] else " [ORT]"), (r[3], r[2]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax[0].set_xlabel("latency, batch 1, 1 thread (ms)"); ax[1].set_xlabel("size on disk (KB, log scale)"); ax[1].set_xscale("log")
    for a in ax:
        a.set_ylabel("test accuracy"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RES / "variants.png", dpi=130)

    # deployment choice per platform: best value of the objective among variants meeting the constraints
    best = max(r[2] for r in V)
    key = {"latency_b128_ms": 5, "latency_b1_ms": 4, "size_kb": 3}
    rows = []
    for plat, c in CFG["platforms"].items():
        ok = [r for r in V if best - r[2] <= c["max_acc_drop"] and (c["max_size_kb"] is None or r[3] <= c["max_size_kb"])]
        if not ok:
            rows.append([plat, c["max_acc_drop"], c["max_size_kb"], c["optimise"], "none fits", "", "", ""])
            continue
        choice = min(ok, key=lambda r: (r[key[c["optimise"]]] / (1 if c["optimise"] != "latency_b128_ms" else 128)))
        rows.append([plat, c["max_acc_drop"], c["max_size_kb"], c["optimise"], f"{choice[0]} [{choice[1]}]", choice[2], choice[3], choice[4]])
        print("deploy", plat, "->", choice[0], choice[1])
    write_csv("deployment_choice.csv", ["platform", "max_accuracy_drop_vs_best", "max_size_kb", "minimise",
                                        "chosen_variant", "test_acc", "size_kb", "latency_b1_ms"], rows)


if __name__ == "__main__":
    RES.mkdir(exist_ok=True)
    C.seed_all(CFG["seed"])
    backend = backend_name()
    torch.backends.quantized.engine = backend
    (RES / "versions.txt").write_text(
        f"python {platform.python_version()}\ntorch {torch.__version__}\nonnx {onnx.__version__}\nonnxruntime {ort.__version__}\n"
        f"quantization backend {backend} (supported: {torch.backends.quantized.supported_engines})\n"
        f"machine {platform.machine()}\nthreads {torch.get_num_threads()}\nplatform {platform.platform()}\n")
    train_d, val_d, test_d = C.load_data(CFG)
    models = float_models(train_d, val_d, test_d)
    for k, m in models.items():
        print(k, "test acc", round(C.accuracy(m, *test_d), 4))
    step_manual(models, test_d)
    dyn = step_dynamic(models, test_d, backend)
    sta = step_static(models, train_d, test_d, backend)
    onx = step_onnx(models, test_d)
    step_variants(models, dyn, sta, onx, test_d, backend)
    print("done")
