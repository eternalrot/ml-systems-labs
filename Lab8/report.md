# Lab 8: Quantization and ONNX

**Student:** Pavlo Budko (72591) · seed 42 · Python 3.11.8, torch 2.3.1+cpu, onnxruntime 1.18.0, backend fbgemm, Windows, x86 CPU

## 1. Goal
Take the Fashion-MNIST CNN from Lab 7 (baseline, structured-pruned, distilled student) and compress it further with int8 quantization and ONNX. Compare all variants in one table and choose one for Cloud, Edge, Mobile and TinyML.

## 2. Method
- Manual symmetric quantization of the weights: `scale = max|W| / (2^(b-1)-1)`, for 8, 4 and 2 bits, per-tensor and per-channel, on `fc1` and `conv3`. I measured MSE and also the accuracy of the whole model with quantized weights.
- PyTorch dynamic quantization (Linear layers) and static quantization (500 random training images for calibration, conv+relu fused).
- A calibration study: different calibration sets, including bad ones.
- ONNX export (opset 17), max difference to PyTorch on 1000 images (tolerance 1e-4), ONNX Runtime latency.
- Latency: median, batch 1 on 1 thread and batch 128, warm-up 50. Details in `config.yaml`, code in `src/lab8.py`.

## 3. Results

**Manual quantization (fc1 weights, MSE):**

| bits | per-tensor | per-channel |
|---|---|---|
| 8 | 1.0e-06 | 2.8e-07 |
| 4 | 2.8e-04 | 9.2e-05 |
| 2 | 2.4e-03 | 2.1e-03 |

Whole-model test accuracy (float 0.9152): 8-bit 0.9155 / 0.9155, 4-bit 0.9005 (per-tensor) / 0.9106 (per-channel), 2-bit 0.1207 / 0.198.

**Calibration study (static int8, baseline):** 500 random images 0.9147–0.9157 (3 seeds), 10 images 0.9142, one class only 0.9148–0.9156, noise 0.9141. Bad range (images x0.1) 0.2978, blank images 0.8704.

**ONNX:** max difference to PyTorch 1.5e-05 (baseline), 7.6e-06 (pruned, distilled), below 1e-4. All argmax equal on 1000 images.

**All variants** (`results/variants.csv`, plot `results/variants.png`):

| variant | runtime | acc | size KB | lat b1 ms | lat b128 ms |
|---|---|---|---|---|---|
| baseline fp32 | PyTorch | 0.9152 | 4028 | 0.339 | 5.38 |
| pruned 50% fp32 | PyTorch | 0.9132 | 2000 | 0.227 | 2.71 |
| distilled fp32 | PyTorch | 0.9043 | 416 | 0.259 | 4.03 |
| baseline fp32 | ONNX RT | 0.9152 | 4027 | 0.127 | 1.34 |
| pruned fp32 | ONNX RT | 0.9132 | 1999 | 0.076 | 0.70 |
| distilled fp32 | ONNX RT | 0.9043 | 415 | 0.049 | 0.58 |
| baseline dynamic int8 | PyTorch | 0.9147 | 1080 | 0.451 | 4.90 |
| baseline static int8 | PyTorch | 0.9157 | 1024 | 0.471 | 3.11 |
| pruned + static int8 | PyTorch | 0.9134 | 516 | 0.374 | 2.10 |
| distilled + static int8 | PyTorch | 0.9050 | 113 | 0.320 | 2.31 |

**Choice per platform** (`results/deployment_choice.csv`):

| platform | limit | chosen variant | acc | size KB | latency |
|---|---|---|---|---|---|
| Cloud | drop <= 1 pp | pruned fp32, ONNX Runtime | 0.9132 | 1999 | 0.70 ms (b128) |
| Edge | drop <= 1.5 pp, <= 5 MB | distilled fp32, ONNX Runtime | 0.9043 | 415 | 0.049 ms (b1) |
| Mobile | drop <= 2 pp, <= 2 MB | distilled fp32, ONNX Runtime | 0.9043 | 415 | 0.049 ms (b1) |
| TinyML | drop <= 3 pp, <= 256 KB | distilled + static int8 | 0.9050 | 113 | 0.32 ms (b1) |

Caveat: I measured on an x86 PC with fbgemm. On a phone (ARM) or a microcontroller the int8 kernels are different, so the latency ranking can change. Size and accuracy would stay the same.

## 4. Conclusions
1. Per-channel quantization is better than per-tensor, and the gap grows when there are fewer bits: at 4 bits the model drops 1.5 pp per-tensor but only 0.5 pp per-channel. 8 bits is basically free (accuracy even went up by 0.03 pp, which is noise), and 2 bits destroys the model (12-20% accuracy).
2. In my measurements int8 saved size (about 4x smaller) but did not make this small CNN faster. Dynamic and static int8 in PyTorch were slower than fp32 at batch 1, and ONNX Runtime fp32 was the fastest by far (about 2.5-5x faster than PyTorch). So for this model, quantization is a tool for size (TinyML), and the runtime matters more for speed.
3. ONNX export is safe here (difference around 1e-5, same predictions). Calibration did not matter much when the images were real, noisy or one class only, but with a wrong range (x0.1) accuracy fell to 30%. So the calibration data must cover the real range of activations, even though it does not have to be big (10 images were enough here).
