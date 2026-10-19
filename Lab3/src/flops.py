"""Hand-calculated FLOPs (1 multiply-add = 2 FLOPs; biases, activations, softmax, LayerNorm ignored)."""


def linear(in_f, out_f, tokens=1):
    return 2 * tokens * in_f * out_f


def conv2d(k, c_in, c_out, h_out, w_out):
    return 2 * k * k * c_in * c_out * h_out * w_out


def attention_layer(seq, d, ff):
    """One Transformer encoder layer (self-attention + feed-forward)."""
    qkv = 2 * seq * d * 3 * d              # Q,K,V projections
    scores = 2 * seq * seq * d             # Q @ K^T (all heads together)
    context = 2 * seq * seq * d            # softmax(scores) @ V
    out_proj = 2 * seq * d * d
    ffn = 2 * seq * d * ff + 2 * seq * ff * d
    return {"qkv": qkv, "scores": scores, "context": context, "out_proj": out_proj,
            "ffn": ffn, "total": qkv + scores + context + out_proj + ffn}


def model_flops():
    mlp = linear(784, 256) + linear(256, 128) + linear(128, 10)
    cnn = conv2d(3, 1, 16, 28, 28) + conv2d(3, 16, 32, 14, 14) + linear(32 * 7 * 7, 10)
    layer = attention_layer(16, 64, 128)["total"]
    attn = linear(49, 64, tokens=16) + 2 * layer + linear(64, 10)
    return {"MLP": mlp, "CNN": cnn, "Attention": attn}


def one_layer_examples():
    a = attention_layer(16, 64, 128)
    return {
        "Linear 784->256": linear(784, 256),
        "Conv2d 16->32, 3x3, 14x14 out": conv2d(3, 16, 32, 14, 14),
        "Attention QKV projections (seq=16, d=64)": a["qkv"],
        "Attention scores QK^T (seq=16, d=64)": a["scores"],
        "Full encoder layer (seq=16, d=64, ff=128)": a["total"],
    }
