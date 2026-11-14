# Lab 7: Compression: pruning and distillation

## 1. Goal
Compress an oversized Fashion-MNIST CNN with pruning and distillation, and measure honestly what each method really gives: accuracy, size, latency, compute and energy.

## 2. Method
- **Baseline.** CNN with 3 conv layers and 2 fully connected layers, 1 030 346 parameters, trained 10 epochs (Adam 1e-3, seed 42). It is also the teacher.
- **Unstructured pruning.** Weights ranked by absolute value and the smallest ones masked at 0, 25, 50, 75, 90, 95%, either globally (all layers ranked together) or layer-wise (same share in every layer). Then 3 epochs of fine-tuning with the mask re-applied after every step; I checked that the final sparsity equals the target.
- **Structured pruning.** In every conv layer the 10, 25 and 50% of filters with the smallest L2 norm are removed and a physically smaller dense model is built, then fine-tuned 3 epochs.
- **Distillation.** Student with 105 866 parameters, loss = 0.5 * CE + 0.5 * T^2 * KL with T = 4 (the standard form, KL of the teacher distribution against the student), trained with and without the teacher for 3 seeds (42, 43, 44), 10 epochs.
- **Measurements.** Latency is the median of 300 runs for batch 1 (1 thread) and of 30 runs for batch 128 (all threads). Sparse size is a CSR file (float32 values, int32 indices). Energy is not measured: it is the CPU time multiplied by an assumed 15 W, so only the comparison between models means something. Machine: Windows, 16 logical CPUs. Code in `src/`, numbers in `results/`.

## 3. Results
**Baseline:** test accuracy 0.9152, 4028 KB, 0.41 ms (batch 1).

**Accuracy against sparsity** (`results/unstructured_pruning.csv`, `structured_pruning.csv`, `sparsity_curves.png`), test accuracy after pruning and after 3 epochs of fine-tuning:

| Removed | Global | Layer-wise | Structured (filters) |
|---|---|---|---|
| 0% (control, only fine-tuning) | 0.9182 | 0.9182 | - |
| 10% filters (9.6% of parameters) | - | - | 0.8904 / 0.9194 |
| 25% | 0.9157 / 0.9190 | 0.9149 / 0.9202 | 0.8402 / 0.9180 (25% filters) |
| 50% | 0.9132 / 0.9202 | 0.8135 / 0.9216 | 0.7443 / 0.9132 (50% filters) |
| 75% | 0.9074 / 0.9204 | 0.3886 / 0.9140 | - |
| 90% | 0.8593 / 0.9178 | 0.1665 / 0.8754 | - |
| 95% | 0.6417 / 0.9133 | 0.1432 / 0.8464 | - |

Each cell is "before fine-tuning / after" (the 0% row is only after). The extra 3 epochs alone give +0.3 points (0.9152 to 0.9182), so small gains over the baseline are not an effect of pruning.

**Latency and size** (`latency_size.png`):

| Model | Latency batch 1 | Latency batch 128 | File, dense | File, CSR |
|---|---|---|---|---|
| baseline | 0.41 ms | 5.8 ms | 4028 KB | 8055 KB |
| global 50% | 0.42 ms | 5.9 ms | 4028 KB | 4032 KB |
| global 90% | 0.40 ms | 5.8 ms | 4028 KB | 813 KB |
| global 95% | 0.41 ms | 5.8 ms | 4028 KB | 411 KB |
| structured 25% filters | 0.31 ms | 4.3 ms | 3008 KB | - |
| structured 50% filters | 0.23 ms | 2.9 ms | 2000 KB | - |

Why unstructured pruning does not speed up the dense CPU kernels: a dense convolution or matrix product walks through every element in regular order and uses vector instructions, and a zero costs the same as any other number. The model is still the same shapes, so the work is the same. To skip the zeros one needs an index for every non-zero weight, and this irregular memory access is slower than the multiplications it saves. I tested this on the first fully connected layer (`results/sparse_kernel_fc1.csv`): a CSR matrix-vector product was 8 times slower than dense at 51% sparsity, 4 times slower at 76%, 1.6 times slower at 91% and only 1.09 times faster at 95%. The CSR file also needs an index per weight, so it is larger than the dense file below 50% sparsity and equal at 50%.

**Distillation** (`results/distillation.csv`, mean of 3 seeds):

| Student (105 866 params, 416 KB) | Test accuracy | Seeds (42, 43, 44) | Latency batch 1 |
|---|---|---|---|
| labels only | 0.9109 +- 0.0007 | 0.9100, 0.9113, 0.9115 | 0.26 ms |
| with teacher (T=4, alpha=0.5) | 0.9027 +- 0.0023 | 0.9041, 0.9046, 0.8994 | 0.28 ms |

**Summary by efficiency pillar** (`results/efficiency_pillars.csv`, energy at an assumed 15 W):

| Model | Data: samples seen | Model: non-zero weights / size | Compute: MACs per image, latency | Energy: inference, training | Accuracy |
|---|---|---|---|---|---|
| baseline | 550 k | 1.03 M / 4028 KB | 2.93 M, 0.41 ms | 681 uJ, 1360 J | 0.9152 |
| global pruning 90% + FT | 715 k | 0.10 M / 813 KB (CSR) | 0.79 M non-zero (2.93 M executed), 0.40 ms | 685 uJ, 1729 J | 0.9178 |
| layer-wise pruning 90% + FT | 715 k | 0.10 M / 813 KB (CSR) | 0.29 M non-zero (2.93 M executed), 0.41 ms | 683 uJ, 1723 J | 0.8754 |
| structured pruning 50% + FT | 715 k | 0.51 M / 2000 KB | 1.01 M, 0.23 ms | 336 uJ, 1579 J | 0.9132 |
| student, labels only | 550 k | 0.11 M / 416 KB | 1.12 M, 0.27 ms | 465 uJ, 796 J | 0.9100 |
| student + distillation | 1100 k (incl. teacher) | 0.11 M / 416 KB | 1.12 M, 0.26 ms | 506 uJ, 2221 J | 0.9041 |

## 4. Conclusions
1. Fewer non-zero weights do not mean a faster model. With global pruning I removed 90% of the weights and the accuracy stayed at 0.918, but the latency stayed at 0.40 ms and the dense file stayed at 4028 KB, because the dense kernels still compute all the zeros. The size only fell when I saved the weights in a sparse format (813 KB), and the sparse kernel was only faster than the dense one at 95% sparsity. Structured pruning, which really makes the layers smaller, was what gave a speed-up (0.23 ms, 1.7 times faster) and a smaller file at 50% of the filters, with a loss of about 0.2 points after fine-tuning.
2. Where the weights are removed matters, and fine-tuning matters. Layer-wise pruning forces the same share in every layer, including the small first convolution layer, and at 90% it dropped to 17% accuracy before fine-tuning and only recovered to 0.875, while global pruning let the big fully connected layer lose most of the weights and still reached 0.918. Fine-tuning fixed most of the damage in all cases (for example global 95%: 0.64 to 0.913), but the control without any pruning also gained 0.3 points from the extra epochs, so I should not count that gain as a result of pruning.
3. Distillation did not help in my experiment: the student trained only on labels got 0.9109 and the distilled one 0.9027, and the distilled student was lower for all three seeds, while costing more training (teacher included). I did not test why. It may be that the teacher (0.915) is not much better than the student, or that T = 4 and alpha = 0.5 are not good for this task, since I used the values from the assignment without tuning. What I take from it is that a technique has to be measured on my own task and not assumed to work; here the simple small student (0.91 accuracy, 10% of the parameters, 0.27 ms) was the cheapest good model.

## Notes
- One seed for pruning and baseline; 3 seeds for distillation only. Differences of a few tenths of a point are noise (the distilled seeds spread by 0.5 points).
- Latency differences of 5% are within measurement noise; the structured speed-ups are much larger.
- Energy values are estimates from CPU time at an assumed power and show relative order only.
