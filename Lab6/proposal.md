# Individual Project Proposal

**Student:** Pavlo Budko  **Student ID:** 72591  **Course:** ML Systems, Vistula University

## Title
A CIFAR-10 image classifier compressed to run on a laptop CPU in under 50 ms and under 5 MB

## Problem
Image classifiers that reach high accuracy (a ResNet-18 has about 11 million parameters, roughly 45 MB in float32) are too large for an edge or mobile application that must answer instantly without a GPU or a server. The project asks how much accuracy has to be paid to make such a model fit a strict budget, and which optimization gives the best accuracy for that budget.

## Dataset
CIFAR-10 (Krizhevsky, 2009): 60 000 colour images of 32x32 pixels in 10 balanced classes (50 000 train, 10 000 test). I hold out 5 000 training images for validation and never use the test set for choices. The data is public, small enough for a laptop, and the classes are balanced, so accuracy is a meaningful metric.

## Success metric
Primary: top-1 accuracy on the CIFAR-10 test set of the final compressed model of **at least 88%**, and not more than 3 percentage points below the teacher. Secondary, reported in one table for every model: accuracy, model file size, median and p95 latency for one image, and accuracy per MB. The comparison is repeated over 3 seeds and reported as mean and range, because a single run is within noise.

## Deployment constraints
- Latency: **at most 50 ms** per image at batch size 1 (p95), measured on a laptop CPU with ONNX Runtime after warm-up.
- Size: **at most 5 MB** for the deployed model file.
- CPU only, no GPU, no network call at inference time.

## Planned optimization
1. **Baseline:** a teacher (ResNet-18 adapted to 32x32) and a small student CNN of roughly 0.3-1 million parameters, both trained with the Lab 6 pipeline (YAML config, MLflow, checkpoints).
2. **Knowledge distillation** from the teacher into the student (soft targets with temperature, mixed with the hard-label loss).
3. **Post-training int8 quantization** of the student (ONNX Runtime) to reduce size and latency.
Pruning is kept as an optional extra if time allows. An ablation shows the effect of each step separately: student alone, student + distillation, student + distillation + int8.

## Plan and risks
Weeks 1-2: teacher and student baselines; weeks 3-4: distillation and quantization; week 5: latency and size measurements, ablation, report. Main risk is that training the teacher on a CPU is slow; mitigation is a shorter schedule, or caching the teacher's outputs once and reusing them. If the 88% target is not reachable within the budget, I will report the best accuracy that fits both constraints and say clearly which constraint limited it.
