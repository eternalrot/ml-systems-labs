import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_data(seed=42):
    X, y = load_digits(return_X_y=True)
    X = X / 16.0                       # pixels are 0..16
    Y = np.eye(10)[y]                  # one-hot
    X_tr, X_te, Y_tr, Y_te, y_tr, y_te = train_test_split(
        X, Y, y, test_size=0.30, stratify=y, random_state=seed)
    return X_tr, X_te, Y_tr, Y_te, y_tr, y_te
