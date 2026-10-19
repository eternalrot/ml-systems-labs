"""Three comparable Fashion-MNIST models: MLP, CNN, small Transformer (attention)."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class AttentionNet(nn.Module):
    """28x28 image -> 16 patches of 7x7 -> Linear(49,64) -> 2 Transformer encoder layers
    (d_model=64, nhead=4, dim_feedforward=128) -> mean pool -> Linear(64,10).
    A learnable positional embedding is added (the manual does not forbid it; without it
    the model cannot tell where a patch came from)."""

    def __init__(self, d_model=64, nhead=4, ff=128, layers=2):
        super().__init__()
        self.proj = nn.Linear(49, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 16, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 10)

    def forward(self, x):
        b = x.shape[0]
        # (B,1,28,28) -> (B,16,49): 4x4 grid of 7x7 patches
        p = x.unfold(2, 7, 7).unfold(3, 7, 7)            # B,1,4,4,7,7
        p = p.reshape(b, 16, 49)
        z = self.proj(p) + self.pos
        z = self.encoder(z)
        return self.head(z.mean(dim=1))


def build_models():
    return {"MLP": MLP(), "CNN": CNN(), "Attention": AttentionNet()}
