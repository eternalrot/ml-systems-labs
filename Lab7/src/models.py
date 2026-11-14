"""Baseline (about 1.03 M parameters) and student (about 106 k parameters) CNNs for Fashion-MNIST."""
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    """conv1 -> pool -> conv2 -> pool -> [conv3] -> flatten -> fc1 -> fc2. Images are 28x28, so the feature map
    entering the classifier is 7x7. channels has 2 entries (student) or 3 entries (baseline)."""

    def __init__(self, channels=(16, 32, 64), hidden=320):
        super().__init__()
        self.channels, self.hidden = tuple(channels), hidden
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1)
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1) if len(channels) == 3 else None
        self.fc1 = nn.Linear(channels[-1] * 49, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        if self.conv3 is not None:
            x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(x)

    def layers(self):
        """Prunable layers in order (weights only; biases are never pruned)."""
        names = ["conv1", "conv2"] + (["conv3"] if self.conv3 is not None else []) + ["fc1", "fc2"]
        return {n: getattr(self, n) for n in names}

    def macs(self, sparse=False):
        """Multiply-accumulates per image; with sparse=True only non-zero weights are counted."""
        pos = {"conv1": 28 * 28, "conv2": 14 * 14, "conv3": 7 * 7, "fc1": 1, "fc2": 1}
        total = 0
        for n, layer in self.layers().items():
            w = layer.weight
            count = int((w != 0).sum()) if sparse else w.numel()
            total += count * pos[n]
        return total


def baseline():
    return CNN((16, 32, 64), 320)


def student():
    return CNN((16, 32), 64)
