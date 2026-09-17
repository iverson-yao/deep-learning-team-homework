# Lightweight CIFAR-100 Image Classification Network

Team deep-learning project exploring lightweight image classification through combinations of ConvNeXt, ECA attention, and Ghost modules.

**Tech Stack:** Python · PyTorch · torchvision · timm · CIFAR-100

**Highlights**
- Built and evaluated multiple lightweight model variants combining ConvNeXt, ECANet, GhostNet-style modules, and ResNet baselines.
- Achieved **79.45% accuracy on CIFAR-100** in the final reported model.
- Reduced parameter count by nearly **50%** relative to the comparison baseline while keeping the reported accuracy gap to **1.24 percentage points**.
- My implementation files are collected under `Zhiyao-Yang/`.

## Project Summary

The project investigates how architectural components from ConvNeXt, efficient channel attention, and Ghost-style feature generation can be combined to improve the accuracy-efficiency tradeoff for CIFAR-100 image classification. Several model variants were implemented and compared, including ConvNeXt-based, ECA-based, Ghost-based, and fused architectures.

## Architecture and Training

The implementation uses PyTorch and includes:

- ConvNeXt-style blocks and model variants.
- Efficient Channel Attention (ECA) components.
- Ghost-based lightweight feature-generation modules.
- CIFAR-100 data loading and augmentation.
- Training utilities including MixUp, CutMix, label smoothing, and cosine/warm-up learning-rate scheduling.
- Validation utilities for comparing candidate architectures.

## Model Variants

Representative implementations in my project directory include:

- `Zhiyao-Yang/convnext_eca_ghost.py`: ConvNeXt + ECA + Ghost fusion experiments.
- `Zhiyao-Yang/convnext_ghost.py`: ConvNeXt + Ghost variant.
- `Zhiyao-Yang/eca+ghost.py`: ECA + Ghost variant.
- `Zhiyao-Yang/ghost_resnet.py`: Ghost-enhanced ResNet variant.
- `Zhiyao-Yang/resnet.py`: ResNet comparison implementation.
- `Zhiyao-Yang/validate.py`: Model validation and evaluation utilities.

## Results

The final reported model reached **79.45% CIFAR-100 accuracy**. In the comparison summarized on my resume, the lightweight design used nearly **50% fewer parameters** than the reference baseline, with an **accuracy difference of 1.24 percentage points**.

These numbers are presented as the project’s reported comparison rather than as a claim that every implemented baseline was outperformed.

## My Contributions

- Designed and implemented lightweight image-classification variants combining ConvNeXt, ECA attention, and Ghost modules in PyTorch.
- Ran architecture and training experiments on CIFAR-100, including data augmentation and optimization strategies, and compared accuracy-efficiency tradeoffs across model variants.
- Implemented and maintained the model and validation code collected under `Zhiyao-Yang/`, including fused architectures and baseline comparisons.
- Contributed to team-level evaluation and final result analysis used to select the reported lightweight model.

## Repository Structure

```text
Convnext/        # ConvNeXt experiment notebooks
ECAnet/          # ECA experiment notebooks
Zhiyao-Yang/     # My model variants and validation code
README.md
```

## Running

The code is written in Python with PyTorch, torchvision, timm, NumPy, and matplotlib. A typical environment should include those dependencies before running the model scripts or notebooks.

For example:

```bash
python Zhiyao-Yang/convnext_eca_ghost.py
```

Training settings and output paths are defined in the individual experiment files.

## Team Scope

This was a team deep-learning project. The repository contains work from multiple team members; the `My Contributions` section and `Zhiyao-Yang/` directory identify the parts I personally focused on.
