"""Quantizable (eager mode) twin of models.CNN: explicit ReLU / pooling modules and Quant/DeQuant stubs."""
import torch
from torch import nn
from torch.ao.quantization import DeQuantStub, QuantStub, convert, fuse_modules, get_default_qconfig, prepare


class QCNN(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.channels, self.hidden = tuple(channels), hidden
        self.quant, self.dequant = QuantStub(), DeQuantStub()
        self.conv1 = nn.Conv2d(1, channels[0], 3, padding=1); self.relu1 = nn.ReLU(); self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, padding=1); self.relu2 = nn.ReLU(); self.pool2 = nn.MaxPool2d(2)
        if len(channels) == 3:
            self.conv3 = nn.Conv2d(channels[1], channels[2], 3, padding=1); self.relu3 = nn.ReLU()
        else:
            self.conv3 = None
        self.fc1 = nn.Linear(channels[-1] * 49, hidden); self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = self.quant(x)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        if self.conv3 is not None:
            x = self.relu3(self.conv3(x))
        x = self.relu4(self.fc1(x.flatten(1)))
        return self.dequant(self.fc2(x))


def static_quantize(float_model, calib_images, backend):
    """Post-training static quantization: fuse conv+relu, insert observers, run calibration images through the
    float model to record activation ranges, then convert weights and activations to int8."""
    torch.backends.quantized.engine = backend
    q = QCNN(float_model.channels, float_model.hidden)
    q.load_state_dict(float_model.state_dict())
    q.eval()
    pairs = [["conv1", "relu1"], ["conv2", "relu2"]] + ([["conv3", "relu3"]] if q.conv3 is not None else []) + [["fc1", "relu4"]]
    fuse_modules(q, pairs, inplace=True)
    q.qconfig = get_default_qconfig(backend)
    prepare(q, inplace=True)
    with torch.no_grad():
        for i in range(0, len(calib_images), 100):
            q(calib_images[i:i + 100])
    convert(q, inplace=True)
    return q
