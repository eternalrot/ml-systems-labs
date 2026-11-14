"""Magnitude pruning (global, layer-wise) and structured filter pruning."""
import torch

from models import CNN


def prunable(model):
    return {n: l.weight for n, l in model.layers().items()}


def global_masks(model, sparsity):
    """Rank all weights of all layers by |w| together and mask the smallest `sparsity` share."""
    ws = prunable(model)
    if sparsity <= 0:
        return {n: torch.ones_like(w) for n, w in ws.items()}
    allw = torch.cat([w.detach().abs().flatten() for w in ws.values()])
    k = int(sparsity * allw.numel())
    thr = torch.kthvalue(allw, k).values
    return {n: (w.detach().abs() > thr).float() for n, w in ws.items()}


def layer_masks(model, sparsity):
    """Same sparsity in every layer: mask the smallest `sparsity` share of each layer separately."""
    out = {}
    for n, w in prunable(model).items():
        if sparsity <= 0:
            out[n] = torch.ones_like(w)
            continue
        k = int(sparsity * w.numel())
        thr = torch.kthvalue(w.detach().abs().flatten(), k).values
        out[n] = (w.detach().abs() > thr).float()
    return out


@torch.no_grad()
def apply_masks(model, masks):
    for n, w in prunable(model).items():
        w.mul_(masks[n])


def sparsity_of(model):
    ws = prunable(model)
    zeros = sum(int((w == 0).sum()) for w in ws.values())
    total = sum(w.numel() for w in ws.values())
    return zeros / total, total - zeros, total


@torch.no_grad()
def structured_prune(model, fraction):
    """Remove the `fraction` of filters with the smallest L2 norm in every conv layer and build a physically
    smaller dense model (next layer loses the matching input channels / fc1 loses the matching columns)."""
    convs = [model.conv1, model.conv2, model.conv3]
    keep = []
    for c in convs:
        norms = c.weight.flatten(1).norm(dim=1)
        n_keep = max(1, int(round(c.out_channels * (1 - fraction))))
        keep.append(torch.sort(torch.topk(norms, n_keep).indices).values)
    new = CNN(tuple(len(k) for k in keep), model.hidden)
    new.conv1.weight.copy_(convs[0].weight[keep[0]]); new.conv1.bias.copy_(convs[0].bias[keep[0]])
    new.conv2.weight.copy_(convs[1].weight[keep[1]][:, keep[0]]); new.conv2.bias.copy_(convs[1].bias[keep[1]])
    new.conv3.weight.copy_(convs[2].weight[keep[2]][:, keep[1]]); new.conv3.bias.copy_(convs[2].bias[keep[2]])
    w = model.fc1.weight.view(model.hidden, convs[2].out_channels, 49)[:, keep[2], :].reshape(model.hidden, -1)
    new.fc1.weight.copy_(w); new.fc1.bias.copy_(model.fc1.bias)
    new.fc2.load_state_dict(model.fc2.state_dict())
    return new
