# OPTIMA: Optimizing Data Augmentation through Bayesian Model Selection

This repository contains the implementation of the OPTIMA framework as described in the paper "Optimizing Data Augmentation through Bayesian Model Selection". OPTIMA provides a novel approach to optimize data augmentation parameters using Bayesian principles, eliminating the need for manual tuning or expensive cross-validation.

## Overview

Data Augmentation (DA) has become an essential tool to improve robustness and generalization in modern machine learning. However, choosing appropriate DA parameters is typically left to trial-and-error or expensive optimization based on validation performance. OPTIMA counters these limitations by taking a probabilistic view of DA, treating augmentation parameters as model hyperparameters and optimizing them through Bayesian model selection.

The framework derives a tractable Evidence Lower BOund (ELBO), allowing for joint optimization of augmentation parameters with model parameters. This approach has been shown to improve calibration and yield robust performance compared to fixed or no augmentation strategies.

## Paper

For a detailed explanation of the methodology, theoretical analysis, and experimental results, please refer to our paper:
[Optimizing Data Augmentation through Bayesian Model Selection](https://arxiv.org/abs/XXXX.XXXXX)

## Repository Structure

- `src/`: Contains the implementation of various data augmentation methods
  - `imagenet_mixup.py`: Implementation of learnable and fixed Mixup augmentation for ImageNet
  - `imagenet_cutmix.py`: Implementation of learnable and fixed CutMix augmentation for ImageNet
  - `imagenet_augmix.py`: Implementation of learnable AugMix with JSD loss for ImageNet

## Implementation Details

The code is implemented using PyTorch and PyTorch Lightning, providing a clean and modular structure. Each augmentation method is implemented as an `Augmenter` class that inherits from `nn.Module`, making it easy to integrate with existing PyTorch models.

### Key Features

- **Learnable Augmentation Parameters**: Instead of fixed hyperparameters, the augmentation parameters are learned during training.
- **Bayesian Framework**: The optimization is grounded in Bayesian principles, providing theoretical guarantees.
- **ImageNet-C Evaluation**: The code includes evaluation on ImageNet-C for robustness testing.
- **Wandb Integration**: Training progress can be monitored using Weights & Biases.

## Usage

Before running the code, update the dataset paths in each file:

```python
# Update these paths to your local setup
IMAGENET_TRAIN = "/path/to/imagenet/train"
IMAGENET_VAL = "/path/to/imagenet/val"
IMAGENET_C = "/path/to/imagenet/ImageNet-C"
```

### Running Experiments

Each script can be run with different augmentation types:

#### Mixup

```bash
python src/imagenet_mixup.py --augmentation_type learnable_mixup --batch_size 128 --learning_rate 0.001
```

#### CutMix

```bash
python src/imagenet_cutmix.py --augmentation_type learnable_cutmix --batch_size 128 --learning_rate 0.001
```

#### AugMix

```bash
python src/imagenet_augmix.py --augmentation_type learnable_augmix_jsd --batch_size 128 --learning_rate 0.001
```

## Requirements

- PyTorch
- PyTorch Lightning
- torchvision
- numpy
- wandb (for logging)

## Citation

If you use this code in your research, please cite our paper:

```
@article{matymov2023optimizing,
  title={Optimizing Data Augmentation through Bayesian Model Selection},
  author={Matymov, Madi and Kampffmeyer, Michael and Tran, Ba-Hien and Heinonen, Markus and Filippone, Maurizio},
  journal={Preprint},
  year={2023}
}
```

## License

[MIT License](LICENSE)
