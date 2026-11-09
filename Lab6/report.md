# Lab 6: Training pipelines and project proposal

## 1. Goal
Make the training of the best Lab 3 model (the CNN, test accuracy 0.8955) a reproducible YAML-driven pipeline with MLflow, and measure what limits training: data loading, batch size, learning-rate schedule, optimizer, gradient accumulation and resuming. The project proposal is in `proposal.md`.

## 2. Method
- `src/train.py` reads `config.yaml` (seed 42, data, model, optimizer, scheduler, checkpoint, MLflow); values can be overridden with `--set key=value`. After every epoch it saves a checkpoint (weights, optimizer, scheduler, epoch, history, random states) and logs parameters, metrics and artifacts to MLflow. Every run of a study is a separate process.
- `src/studies.py` runs all studies on a 20 000-image training subset with Fashion-MNIST augmentation done per sample in the DataLoader, Adam 1e-3, one seed. Workers: 3 epochs, median of epochs 2-3. Batch size: 5 epochs. Schedules and optimizers: 10 epochs. Peak memory is the resident memory of the main process.
- Profiling uses `torch.profiler` over one epoch with the phases labelled, next to wall-clock timers. Machine: Windows, 16 logical CPUs, 10 torch threads, CPU only.

## 3. Results
**Profile of one epoch** (`results/profile_breakdown.csv`):

| Phase | 0 workers | 2 workers |
|---|---|---|
| data loading | 75.6% | 25.2% |
| forward | 9.5% | 28.7% |
| backward | 12.4% | 37.0% |
| optimizer step | 2.5% | 9.1% |

**Workers** (`results/workers.csv`, `workers.png`):

| num_workers | 0 | 1 | 2 | 4 | 8 | compute only |
|---|---|---|---|---|---|---|
| samples/s | 4126 | 5237 | 9041 | 11329 | 11452 | 12603 |
| waiting for data | 70% | 61% | 30% | 6% | 6% | 9% |

**Batch size, 5 epochs** (`results/batch_size.csv`):

| Batch | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| samples/s | 6414 | 8319 | 8727 | 9001 | 9016 |
| peak memory (MB) | 556 | 555 | 554 | 560 | 565 |
| test accuracy | 0.8430 | 0.8302 | 0.8121 | 0.8152 | 0.7883 |

**Schedules and optimizers, 10 epochs** (`results/schedules_optimizers.csv`):

| Configuration | Best val. acc. | Test acc. |
|---|---|---|
| Adam, constant lr | 0.8452 | 0.8375 |
| Adam + StepLR | 0.8352 | 0.8362 |
| Adam + CosineAnnealingLR | 0.8372 | 0.8377 |
| Adam + ReduceLROnPlateau (patience 0) | 0.8456 | 0.8446 |
| SGD + momentum (lr 0.05) + CosineAnnealingLR | 0.8736 | 0.8728 |

**Gradient accumulation** (`results/grad_accumulation.csv`): batch 256 gives train loss 0.623 and test accuracy 0.8152; batch 64 x 4 gives 0.616 and 0.8031; batch 64 without accumulation gives 0.510 and 0.8302. For the same 256 images the gradients of one batch of 256 and of four accumulated batches of 64 differ by at most 2.9e-8.

**Interrupt and resume** (`results/resume.csv`): training stopped after epoch 3 and continued from `last.pt` to epoch 6. The difference from an uninterrupted run is 0.0 in training loss and validation accuracy at every epoch, and the test accuracy is the same (0.8419).

All runs are in the MLflow experiment `lab6` (`results/mlflow_runs.csv`).

## 4. Conclusions
1. At the start my training was slow mostly because the CPU was busy preparing images, not because the network was heavy: with one process it waited for data 70% of the time. When I added workers the speed went from 4126 to about 11 300 samples/s, and after 4 workers nothing improved any more, so from that point the model itself was the limit. So before making a model faster I should first look at where the time goes.
2. A bigger batch trains faster, but it is not free. From batch 32 to 256 the speed grew from 6414 to 9001 samples/s and then stopped growing, while the accuracy after 5 epochs fell from 0.843 to 0.788 because the model makes fewer updates. Gradient accumulation lets me get a batch-256 step while only keeping 64 images in memory; I checked that the gradients are the same (difference 3e-8) and the training looked similar.
3. For a run to be reproducible and resumable it is not enough to save the weights: I also saved the optimizer, scheduler and random states, and because of that the run stopped at epoch 3 and continued gave exactly the same numbers as the normal run. I also learned to be careful with small differences: the three Adam schedules differed by less than 1 point with one seed, so I cannot say which is better.

## Notes
- One seed and a 20 000-image subset: differences of a few tenths of a point are noise. Test accuracy was reported but not used for choices.
- ReduceLROnPlateau with patience 1 never reduced the rate in 10 epochs, so patience 0 was used.
- Throughput numbers depend on the machine; a first run on a 2-core machine showed no gain from workers.
