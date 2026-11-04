# Lab 5: Frameworks and automatic differentiation

## 1. Goal
Build a scalar reverse-mode autodiff engine, check its gradients against PyTorch and numerical differences, train a 2-8-1 network with it, and compare it with PyTorch (speed, graph, eager vs compiled, non-contiguous tensors).

## 2. Method
- **Engine (`src/engine.py`).** A `Value` stores `data`, `grad`, its parents and a local derivative rule. `+`, `*`, `**` and ReLU build the graph during the forward pass; I added `exp`, `log` and `sigmoid` for the BCE loss. `backward()` sorts the graph topologically, sets the output gradient to 1, goes through the nodes in reverse and accumulates with `+=`.
- **Gradient check.** `sum(relu((a*b + c)**2))`, vectors of length 8, float64: own engine vs PyTorch autograd vs central differences (h = 1e-6). Two extra cases test accumulation: `x*x + x` and `(x+1)(x+2)`.
- **Moons.** `make_moons(300, noise=0.2)`, stratified 70/30 split, 2 -> 8 (ReLU) -> 1 (sigmoid), BCE, full-batch gradient descent, lr 0.5, 100 epochs, same initial weights in both implementations; median of 5 runs.
- **PyTorch part.** MLP 256-512-512-10: graph inspection, eager vs `torch.jit.trace` vs `torch.compile` after warm-up (median of 5 x 50 calls), and a 2048x2048 matrix vs its transposed view.
- Seed 42 everywhere; versions in `results/versions.txt`; code in `src/`.

## 3. Results
| Check | Result |
|---|---|
| max abs gradient difference, own vs PyTorch | 0.0 |
| max abs gradient difference, own vs numerical | 1.1e-9 (limit 1e-6) |
| accumulation cases (expected 7 and 9) | 7.0 and 9.0 |

| Moons, 100 epochs | Median time | Final loss | Test accuracy |
|---|---|---|---|
| own engine | 8.96 s | 0.2498 | 0.856 |
| PyTorch (float64) | 0.022 s | 0.2498 | 0.856 |

The loss curves differ by at most 7e-16; PyTorch is about 400 times faster.

The autograd graph of the MLP is `Mean -> Logsumexp -> (Addmm -> Relu) x 2 -> Addmm` with `AccumulateGrad` leaves and `T` nodes for the transposed weights (`results/graph_inspection.txt`). `weight.T` is a view with stride (1, 256) and is not contiguous.

| Mode (batch 64) | Time per forward | Speedup |
|---|---|---|
| eager | 502 us | 1.00 |
| `torch.jit.trace` + freeze | 479 us | 1.05 |
| `torch.compile` after warm-up | 508 us | 0.99 |

The first `torch.compile` call took 1.4 s. At batch 1, trace was 1.08x faster than eager.

| 2048x2048 operation | Contiguous | Transposed view | Ratio |
|---|---|---|---|
| relu | 0.417 ms | 0.431 ms | 1.03 |
| sum(dim=1) | 0.421 ms | 0.625 ms | 1.49 |
| matmul | 27.4 ms | 26.2 ms | 0.95 |

## 4. Conclusions
1. The engine is correct because gradients are accumulated with `+=`: a node used twice receives the sum of both contributions. The result equals PyTorch exactly and the numerical estimate to 1e-9, and the separate `x*x + x` test would fail with overwriting.
2. A framework is fast because one operation covers a whole batch. Both implementations give the same losses, but my engine handles one number at a time through Python objects and is 400 times slower.
3. Compilation and contiguity do not always matter. On this small MLP `torch.compile` gave no speedup (0.99x) and trace only 5%, because large matrix multiplications dominate; it pays off with many small fusable operations or a model run very many times. A transposed view slowed only the reduction (1.49x), not relu or matmul.

## Notes
- Timings are from one machine and noisy; differences of a few percent are within noise. A first run on another machine (2 threads, Linux) gave compile 0.76x and matmul on a view 1.36x slower, so only gradients and losses are machine-independent.
- The engine supports scalars and constant exponents only; a strided view such as `x[:, ::2]` was not timed.
