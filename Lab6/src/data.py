"""Fashion-MNIST loaders. Training samples go through a per-sample augmentation pipeline in the DataLoader
(this is the real data-loading cost that num_workers can hide); validation/test are converted once."""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

MEAN, STD = (0.2860,), (0.3530,)


def _to_tensors(ds):
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    return TensorDataset(x, y)


def make_loaders(cfg, root_dir: Path, loader_generator: torch.Generator):
    d, t = cfg["data"], cfg["train"]
    root = (root_dir / d["root"]).resolve()
    norm = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    aug = [transforms.RandomHorizontalFlip(), transforms.RandomAffine(10, translate=(0.1, 0.1))] if d["augment"] else []
    train_full = datasets.FashionMNIST(root=str(root), train=True, download=True,
                                       transform=transforms.Compose(aug + norm))
    eval_full = datasets.FashionMNIST(root=str(root), train=True, download=True,
                                      transform=transforms.Compose(norm))
    test_ds = datasets.FashionMNIST(root=str(root), train=False, download=True,
                                    transform=transforms.Compose(norm))
    g = torch.Generator().manual_seed(cfg["seed"])          # fixed split, independent of other randomness
    perm = torch.randperm(len(train_full), generator=g)
    val_idx = perm[: d["val_size"]].tolist()
    train_idx = perm[d["val_size"]: d["val_size"] + d["train_size"]].tolist()
    nw = int(d["num_workers"])
    if d.get("preload"):                         # whole training set converted once: no per-sample loading cost
        assert not d["augment"], "preload needs augment: false"
        train_set = _to_tensors(Subset(eval_full, train_idx))
    else:
        train_set = Subset(train_full, train_idx)
    train_loader = DataLoader(train_set, batch_size=t["batch_size"], shuffle=True,
                              generator=loader_generator, num_workers=nw, persistent_workers=nw > 0)
    val_loader = DataLoader(_to_tensors(Subset(eval_full, val_idx)), batch_size=1000)
    test_loader = DataLoader(_to_tensors(test_ds), batch_size=1000)
    return train_loader, val_loader, test_loader
