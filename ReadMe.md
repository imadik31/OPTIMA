# Learnable AugMix Severity with JSD for ImageNet Classification

This project implements and evaluates an experimental AugMix variant where the **augmentation severity is a learnable parameter**. This is combined with the standard Jensen-Shannon Divergence (JSD) consistency loss for training a ResNet-50 model on ImageNet.

**Note on Performance:** This "Type 1 Learnable" approach for AugMix, where PIL-based augmentations are processed within an `nn.Module` in the main training loop, can be **significantly slower** than applying fixed augmentations in parallel DataLoader workers. This implementation prioritizes the learnable parameter mechanism over maximum speed.

## Features

-   ResNet-50 architecture.
-   ImageNet training pipeline using PyTorch Lightning.
-   **Learnable AugMix Severity:**
    -   The core AugMix data transformation (multiple augmentation chains, mixing) is applied.
    -   The overall **severity** of the basic augmentations within these chains is controlled by a learnable distribution.
    -   Specifically, it learns parameters (`log_severity_mu`, `log_severity_log_sigma`) of a Normal distribution over `log(aug_severity_param)`.
    -   A `aug_severity_value` is sampled per batch from `exp(Normal(mu, sigma^2))` and used to control the intensity of base augmentations.
    -   Includes a KL divergence term in the loss to regularize the learned distribution of `log(aug_severity_param)`.
-   **Jensen-Shannon Divergence (JSD) Loss:** Encourages consistent predictions across three views of an image (original, AugMix view 1, AugMix view 2).
-   Evaluation on standard ImageNet validation and ImageNet-C.
-   Weights & Biases (WandB) integration.
-   Checkpointing of the best model.
-   Detailed JSON output of results.

## Prerequisites

-   Python 3.8+
-   PyTorch (e.g., 2.0+) with CUDA
-   Torchvision
-   PyTorch Lightning (e.g., 2.0+)
-   NumPy
-   WandB (optional, `pip install wandb`)
-   Pillow

### Datasets
1.  **ImageNet (ILSVRC2012)**
2.  **ImageNet-C**

**IMPORTANT:** Update dataset paths in `learnable_augmix_jsd_v_type1_imagenet.py`:
`IMAGENET_TRAIN`, `IMAGENET_VAL`, `IMAGENET_C`.

## Setup

1.  **Save Script:** Save the Python code.
2.  **Environment (conda recommended):**
    ```bash
    conda create -n laugmix python=3.10
    conda activate laugmix
    ```
3.  **Install Dependencies:**
    PyTorch (e.g., CUDA 11.8):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    Others:
    ```bash
    pip install pytorch-lightning numpy wandb Pillow
    ```
    If using WandB: `wandb login`
4.  **Verify Dataset Paths.**

## How to Train

### Training with Learnable AugMix Severity + JSD
The script defaults to `aug_type="learnable_augmix_jsd"`.
-   `--initial_aug_severity`: Sets the initial mean for the prior of the learnable severity.
-   `--beta_jsd`: Weight for the JSD consistency loss.
-   `--beta_augmenter_reg`: Weight for the KL divergence of the learnable severity parameters.

```bash
python learnable_augmix_jsd_v_type1_imagenet.py \
    --epochs 90 \
    --batch 128 \
    --workers 8 \
    --lr 1e-4 \
    --gpus -1 \
    --aug_type learnable_augmix_jsd \
    --augmix_mixture_width 3 \
    --augmix_mixture_depth -1 \
    --initial_aug_severity 3.0 \
    --prior_severity_std_learnable_aug 1.0 \
    --prior_severity_mean_offset_learnable_aug 0.0 \
    --beta_jsd 12.0 \
    --beta_augmenter_reg 1.0 \
    --beta_net 0.01 \
    --precision "16" \
    --matmul_precision medium \
    --grad_clip_val 1.0 \
    --ckpt_dir_base ./my_learnable_augmix_checkpoints


Note on Batch Size: --batch 128 is used as a default because processing three views for JSD and performing AugMix on-the-fly is memory and compute intensive. Adjust based on your hardware.

## How to Evaluate

1. Evaluate a Trained Learnable AugMix Model
python learnable_augmix_jsd_v_type1_imagenet.py \
    --eval_only \
    --aug_type learnable_augmix_jsd \
    --ckpt_path ./my_learnable_augmix_checkpoints/learnable_augmix_jsd/learnable_augmix_jsd-best-....ckpt \
    --batch 128 \
    --gpus -1 \
    # Pass relevant augmix hparams if not fully stored or if overriding for analysis
    --augmix_mixture_width 3 \
    --augmix_mixture_depth -1 \
    --initial_aug_severity 3.0 \
    --beta_jsd 12.0 \
    --beta_augmenter_reg 1.0


2. Evaluate Torchvision\'s Pretrained ResNet-50 (Baseline)
python learnable_augmix_jsd_v_type1_imagenet.py \
    --eval_only \
    --aug_type none \
    --batch 256 \
    --gpus -1


## Key Command-Line Arguments

--aug_type: learnable_augmix_jsd, none.

--augmix_mixture_width, --augmix_mixture_depth: AugMix structural parameters.

--initial_aug_severity: Initial mean for the distribution of learnable augmentation severity.

--prior_severity_std_learnable_aug, --prior_severity_mean_offset_learnable_aug: Parameters for the prior distribution of the learnable severity.

--beta_jsd: Weight for the JSD consistency loss.

--beta_augmenter_reg: Weight for the KL divergence of the learnable severity parameters.

(Other arguments like --epochs, --batch, --lr, etc., are standard).

## Output

Checkpoints: Best model saved in <ckpt_dir_base>/<aug_type>/.

JSON Results: Detailed metrics.

WandB Logs: Run tracking.

## Learnable AugMix Severity Details

This implementation attempts to make the aug_severity of AugMix a learnable parameter.

It learns log_severity_mu and log_severity_log_sigma for a Normal distribution.

aug_severity for each batch is sampled as exp(Normal(log_severity_mu, exp(log_severity_log_sigma)^2)).

This sampled severity then dictates the maximum intensity of individual PIL-based augmentations within the AugMix chains.

A KL divergence term regularizes the learned distribution of log_severity against a prior.

This is combined with the JSD loss, which encourages prediction consistency across an original view and two AugMix-generated views of an image.

