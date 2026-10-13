"""Two-hidden-layer neural network from scratch (NumPy only).

Architecture: 64 -> W (ReLU) -> W (ReLU) -> 10 (softmax)
Shapes: W1 (64,W), W2 (W,W), W3 (W,10); b1 (W,), b2 (W,), b3 (10,)
"""
import random

import numpy as np

SEED = 42


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:  # required by the course manual; torch is not used in this lab
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def init_params(width, rng, n_in=64, n_out=10):
    """Weights ~ N(0, 0.01) (std = 0.01), biases = 0."""
    return {
        "W1": rng.normal(0.0, 0.01, (n_in, width)), "b1": np.zeros(width),
        "W2": rng.normal(0.0, 0.01, (width, width)), "b2": np.zeros(width),
        "W3": rng.normal(0.0, 0.01, (width, n_out)), "b3": np.zeros(n_out),
    }


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    """Layer-by-layer forward pass. Returns probabilities and a cache."""
    Z1 = X @ p["W1"] + p["b1"]
    A1 = np.maximum(Z1, 0.0)
    Z2 = A1 @ p["W2"] + p["b2"]
    A2 = np.maximum(Z2, 0.0)
    Z3 = A2 @ p["W3"] + p["b3"]
    P = softmax(Z3)
    return P, (X, Z1, A1, Z2, A2)


def cross_entropy(P, Y):
    """Mean cross-entropy; Y is one-hot."""
    return -np.mean(np.sum(Y * np.log(P + 1e-12), axis=1))


def backward(p, cache, P, Y):
    """Analytic gradients of the mean cross-entropy (chain rule, derived by hand)."""
    X, Z1, A1, Z2, A2 = cache
    B = X.shape[0]
    dZ3 = (P - Y) / B                       # softmax + CE combined
    g = {"W3": A2.T @ dZ3, "b3": dZ3.sum(axis=0)}
    dZ2 = (dZ3 @ p["W3"].T) * (Z2 > 0)      # ReLU'
    g["W2"], g["b2"] = A1.T @ dZ2, dZ2.sum(axis=0)
    dZ1 = (dZ2 @ p["W2"].T) * (Z1 > 0)
    g["W1"], g["b1"] = X.T @ dZ1, dZ1.sum(axis=0)
    return g


def loss_only(p, X, Y):
    return cross_entropy(forward(p, X)[0], Y)


def accuracy(p, X, y):
    return float(np.mean(forward(p, X)[0].argmax(axis=1) == y))
