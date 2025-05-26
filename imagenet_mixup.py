#!/usr/bin/env python3
# learnable_mixup_imagenet.py
# ------------------------------------------------------------
#  ImageNet training with *learnable* Mixup alpha, fixed Mixup,
#  or no augmentation (eval only).
#  Lightning-based; comes with ImageNet-C evaluation.
# ------------------------------------------------------------
import os
import math
import random
import json
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
import wandb # Explicit import for safety, though PL might lazy import
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dset
import torchvision.models as models

# ───────────────────────── paths & defaults ──────────────────
# Define default paths for ImageNet datasets and output directories
# !!! USERS SHOULD UPDATE THESE TO THEIR LOCAL SETUP !!!
IMAGENET_TRAIN = "/path/to/ibex/train"
IMAGENET_VAL = "/path/to/ibex/val"
IMAGENET_C = "/path/to/ibex/ImageNet-C" # Base directory for ImageNet-C

PROJECT        = "LearnableMixup_IMNET_Full" # WandB Project Name
CKPT_DIR_LEARN = "./checkpoints_learnable_mixup" # Default for learnable mixup checkpoints
CKPT_DIR_FIXED = "./checkpoints_fixed_mixup"     # Default for fixed mixup checkpoints

# Standard ImageNet normalization constants
MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float)
STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float)

# List of corruption types in ImageNet-C
CORRUPTIONS = [
 "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
 "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
 "elastic_transform","pixelate","jpeg_compression"
]

# ───────────────────────── data loaders ──────────────────────
def loaders(batch_size_arg, num_workers_arg, current_device_type="cpu"): # Added current_device_type for pin_memory
    """
    Creates and returns DataLoaders for ImageNet training and validation.
    Standard augmentations (RandomResizedCrop, RandomHorizontalFlip, ToTensor) are applied.
    """
    # Define standard augmentations for training
    train_transforms = T.Compose([
        T.RandomResizedCrop(224),      # Crop a random portion of image and resize it to 224x224
        T.RandomHorizontalFlip(),      # Horizontally flip the image with 50% probability
        T.ToTensor()                   # Convert PIL Image to PyTorch Tensor (scales to [0,1])
    ])
    # Define standard transformations for validation (no augmentation)
    val_transforms = T.Compose([
        T.Resize(256),                 # Resize to 256x256
        T.CenterCrop(224),             # Crop the center 224x224
        T.ToTensor()                   # Convert PIL Image to PyTorch Tensor
    ])

    # Ensure specified ImageNet paths exist
    if not os.path.isdir(IMAGENET_TRAIN):
         raise FileNotFoundError(f"ImageNet Train directory not found: {IMAGENET_TRAIN}. Please update path.")
    if not os.path.isdir(IMAGENET_VAL):
         raise FileNotFoundError(f"ImageNet Val directory not found: {IMAGENET_VAL}. Please update path.")

    # Create ImageFolder datasets
    train_dataset = dset.ImageFolder(IMAGENET_TRAIN, train_transforms)
    val_dataset = dset.ImageFolder(IMAGENET_VAL,   val_transforms)

    use_persistent_workers = num_workers_arg > 0
    pin_memory_flag = (current_device_type == "gpu") # Use pin_memory only if on GPU

    # Create DataLoaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size_arg,
                              shuffle=True,                  # Shuffle training data each epoch
                              num_workers=num_workers_arg,   # Number of subprocesses for data loading
                              pin_memory=pin_memory_flag,    # If True, copies Tensors into CUDA pinned memory before returning them
                              persistent_workers=use_persistent_workers) # If True, workers are not shutdown after every epoch
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size_arg,
                            shuffle=False,                 # No shuffling for validation
                            num_workers=num_workers_arg,
                            pin_memory=pin_memory_flag,
                            persistent_workers=use_persistent_workers)
    return train_loader, val_loader, val_transforms # Return val_transforms for ImageNet-C

def loaders_imagenet_c(transforms_arg, batch_size_arg, num_workers_arg, current_device_type="cpu"):
    """
    Creates DataLoaders for each corruption and severity level in ImageNet-C.
    """
    output_dataloaders = []
    if not os.path.isdir(IMAGENET_C):
        print(f"Warning: ImageNet-C base directory not found: {IMAGENET_C}. Skipping ImageNet-C.")
        return []

    use_persistent_workers = num_workers_arg > 0
    pin_memory_flag = (current_device_type == "gpu")

    for corruption_name in CORRUPTIONS:
        corruption_type_path = os.path.join(IMAGENET_C, corruption_name)
        if not os.path.isdir(corruption_type_path):
            # print(f"Info: ImageNet-C corruption type path not found: {corruption_type_path}") # Optional info
            continue

        for severity_level in range(1,6): # Severities 1 through 5
            corruption_path = os.path.join(corruption_type_path, str(severity_level))
            if not os.path.isdir(corruption_path):
                # print(f"Warning: ImageNet-C severity path not found: {corruption_path}") # Optional warning
                continue

            corrupted_dataset = dset.ImageFolder(corruption_path, transforms_arg) # Use validation transforms
            corruption_loader = DataLoader(corrupted_dataset,
                                           batch_size=batch_size_arg,
                                           shuffle=False,
                                           num_workers=num_workers_arg,
                                           pin_memory=pin_memory_flag,
                                           persistent_workers=use_persistent_workers)
            output_dataloaders.append((corruption_name, severity_level, corruption_loader))
    
    if not output_dataloaders:
        print(f"Warning: No ImageNet-C data was loaded from {IMAGENET_C}. Check paths and dataset structure.")
    return output_dataloaders

# ─────────────────── Augmenter Base Class ───────────────────
class Augmenter(nn.Module):
    """
    Base class for image augmenters.
    Subclasses should implement the forward method and optionally the kl method.
    """
    requires_labels = False # Default: augmenter does not require labels (e.g., simple transforms)
                            # Mixup/CutMix will override this to True.

    def __init__(self):
        super().__init__()

    def forward(self, x_input, y_input=None): # y_input is optional
        """
        Applies the augmentation.
        Args:
            x_input: Batch of input images (Tensor).
            y_input: Batch of corresponding labels (Tensor, optional).
        Returns:
            Augmented images, and potentially modified/multiple labels and mixing coefficients.
        """
        raise NotImplementedError

    def kl(self):
        """
        Calculates the KL divergence term for learnable augmenters.
        Default is 0 if the augmenter is not learnable or has no specific KL.
        """
        device = 'cpu' # Default device
        # Try to infer device from specific known parameters or any parameter/buffer
        if hasattr(self, 'mu') and isinstance(self.mu, nn.Parameter): device = self.mu.device
        elif hasattr(self, 'log_beta_alpha_mu'): device = self.log_beta_alpha_mu.device # For LearnableCutMix style
        elif hasattr(self, 'log_severity_mu'): device = self.log_severity_mu.device # For LearnableAugMix style
        elif hasattr(self, 'dummy_param') and isinstance(self.dummy_param, torch.Tensor): device = self.dummy_param.device
        else:
            try: device = next(self.parameters()).device
            except StopIteration:
                try: device = next(self.buffers()).device
                except StopIteration: pass
        return torch.tensor(0.0, device=device)

# ─────────────────── No Augmentation Wrapper ──────────────────
class NoAug(Augmenter):
    """
    An augmenter that performs no operation (identity).
    """
    def __init__(self):
        super().__init__()
        # Register a dummy buffer to easily get the module's device for kl()
        self.register_buffer("dummy_param", torch.empty(0))

    def forward(self, x_input, y_input=None):
        return x_input # Returns the input image unchanged

# ─────────────────── Fixed Mixup augmenter ──────────────────
class FixedMixup(Augmenter):
    """
    Implements Mixup augmentation with a fixed alpha hyperparameter.
    Mixes pairs of images and their one-hot encoded labels.
    lambda ~ Beta(alpha, alpha)
    x_mixed = lambda * x1 + (1 - lambda) * x2
    y_mixed = lambda * y1 + (1 - lambda) * y2 (for one-hot labels)
    Since we use cross-entropy with integer labels, the loss is:
    lambda * CE(output, y1) + (1 - lambda) * CE(output, y2)
    """
    requires_labels = True # Mixup needs labels

    def __init__(self, alpha_val=0.2):
        super().__init__()
        if alpha_val <= 0:
            raise ValueError("Alpha for Mixup's Beta distribution must be > 0.")
        self.alpha = alpha_val
        self.register_buffer("dummy_param", torch.empty(0))

    def forward(self, x_input, y_input): # Expects x_input (B,C,H,W), y_input (B,)
        batch_size = x_input.size(0)
        device = x_input.device # Device of input tensors

        # Create Beta distribution parameter on the correct device
        beta_param_alpha = torch.tensor(self.alpha, dtype=torch.float, device=device)
        # Sample lambda coefficients for the batch from Beta(alpha, alpha)
        lambda_coeffs = torch.distributions.Beta(beta_param_alpha, beta_param_alpha).sample((batch_size,)).to(device)
        # Clamp lambda to avoid exact 0 or 1, which can cause issues with loss if reduction='none'
        lambda_coeffs_clamped = torch.clamp(lambda_coeffs, 1e-6, 1.0 - 1e-6)

        # Get a random permutation of indices for mixing
        permutation_indices = torch.randperm(batch_size, device=device)
        
        # Perform Mixup: x_mixed = lambda * x_i + (1-lambda) * x_j
        # lambda_coeffs_clamped needs to be reshaped for broadcasting with image tensors
        # (B,) -> (B,1,1,1)
        x_mixed = lambda_coeffs_clamped.view(-1,1,1,1) * x_input + \
                  (1 - lambda_coeffs_clamped).view(-1,1,1,1) * x_input[permutation_indices]
        
        # Return mixed images, original labels, permuted labels, and lambda coefficients
        return x_mixed, y_input, y_input[permutation_indices], lambda_coeffs_clamped

# ─────────────────── learnable Mixup augmenter ───────────────
class LearnableMixup(Augmenter):
    """
    Implements Mixup augmentation where the alpha hyperparameter for the
    Beta distribution is learned.
    Specifically, it learns mu and log_sigma for a Normal distribution over logit(alpha).
    alpha_i ~ sigmoid(Normal(mu, sigma^2)) for each image/pair in the batch.
    lambda_i ~ Beta(alpha_i, alpha_i)
    """
    requires_labels = True

    def __init__(self, initial_alpha=0.2, prior_standard_dev=2.0, prior_mean_offset_val=0.0):
        super().__init__()
        self.initial_alpha_value = initial_alpha # For reference and prior centering
        self.prior_std_value = prior_standard_dev     # For reference

        # Clamp initial_alpha to (eps, 1-eps) for torch.logit
        clamped_initial_alpha = np.clip(initial_alpha, 1e-6, 1.0 - 1e-6)
        
        # Learnable parameters for the Normal distribution of logit(alpha)
        # mu: mean of logit(alpha)
        self.mu = nn.Parameter(torch.logit(torch.tensor(clamped_initial_alpha, dtype=torch.float)))
        # log_sigma: log of standard deviation of logit(alpha), ensures sigma > 0
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(0.1, dtype=torch.float))) # Start with small variance

        # Fixed prior distribution parameters for logit(alpha)
        # Prior is Normal(prior_mu_logit, prior_std_buffer^2)
        prior_mu_logit = torch.logit(torch.tensor(clamped_initial_alpha, dtype=torch.float)) + prior_mean_offset_val
        self.register_buffer("prior_mu_buffer",  prior_mu_logit)
        self.register_buffer("prior_std_buffer", torch.tensor(prior_standard_dev, dtype=torch.float))

    def forward(self, x_input, y_input):
        batch_size = x_input.size(0)
        device = self.mu.device # Parameters are on the correct device due to nn.Module behavior

        # Sample logit(alpha_i) for each example using reparameterization trick
        # Epsilon ~ Normal(0,1)
        epsilon_noise = torch.randn(batch_size, device=device) # (B,)
        # self.mu and self.log_sigma are scalar parameters. Unsqueeze for broadcasting.
        # logit_alpha_samples will have shape (B,)
        logit_alpha_samples = self.mu.unsqueeze(0) + torch.exp(self.log_sigma).unsqueeze(0) * epsilon_noise
        
        # Transform logit_alpha samples to alpha samples in (0,1)
        alpha_parameters = torch.sigmoid(logit_alpha_samples)
        # Clamp alpha to (eps, 1-eps) for Beta distribution stability
        alpha_parameters_clamped = torch.clamp(alpha_parameters, 1e-6, 1.0 - 1e-6)
        # If batch_size is 1, alpha_parameters_clamped might be scalar, unsqueeze for Beta
        if alpha_parameters_clamped.ndim == 0:
            alpha_parameters_clamped = alpha_parameters_clamped.unsqueeze(0)

        # Sample lambda_i ~ Beta(alpha_i, alpha_i) for each image
        lambda_coeffs = torch.distributions.Beta(alpha_parameters_clamped, alpha_parameters_clamped).sample()
        lambda_coeffs_clamped = torch.clamp(lambda_coeffs, 1e-6, 1.0 - 1e-6)

        permutation_indices = torch.randperm(batch_size, device=device)
        x_mixed = lambda_coeffs_clamped.view(-1,1,1,1) * x_input + \
                  (1 - lambda_coeffs_clamped).view(-1,1,1,1) * x_input[permutation_indices]
        return x_mixed, y_input, y_input[permutation_indices], lambda_coeffs_clamped

    def kl(self):
        """
        Calculates KL divergence between the learned Normal distribution of logit(alpha)
        and the fixed prior Normal distribution.
        KL( N(mu, sigma_q^2) || N(prior_mu, prior_std^2) )
        """
        sigma_q_squared = torch.exp(self.log_sigma * 2) # Learned variance (sigma_q^2)
        prior_std_squared = self.prior_std_buffer**2    # Prior variance

        # KL divergence formula for two Gaussians
        term1 = (sigma_q_squared + (self.mu - self.prior_mu_buffer)**2) / (2 * prior_std_squared)
        term2 = torch.log(self.prior_std_buffer / torch.exp(self.log_sigma)) # log(prior_std / sigma_q)
        kl_divergence = term1 + term2 - 0.5
        return kl_divergence

# ───────────────────── Lightning module ──────────────────────
class LitImageNetModel(pl.LightningModule):
    """
    PyTorch Lightning module for ImageNet classification with different augmentation strategies.
    """
    def __init__(self, learning_rate=1e-4, beta_network_reg=0.01, beta_augmenter_reg=1.0,
                 augmentation_type="learnable_mixup", fixed_mixup_alpha_val=0.2,
                 load_pretrained_for_no_aug_eval=False):
        super().__init__()
        # Save all constructor arguments to self.hparams attribute
        # This also makes them available for ModelCheckpoint and logging.
        self.save_hyperparameters()

        # Initialize network (ResNet50)
        if self.hparams.load_pretrained_for_no_aug_eval:
            print("Loading pretrained ResNet50 (IMAGENET1K_V2) for No Augmentation evaluation.")
            # Load pretrained weights without modifying the final layer (for direct evaluation)
            network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            # Load pretrained weights and replace the final fully connected layer for fine-tuning
            network = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, 1000) # ImageNet has 1000 classes
        self.net = network

        # Initialize the specified augmenter
        if self.hparams.augmentation_type == "learnable_mixup":
            self.augmenter = LearnableMixup(initial_alpha=self.hparams.fixed_mixup_alpha_val)
        elif self.hparams.augmentation_type == "fixed_mixup":
            self.augmenter = FixedMixup(alpha_val=self.hparams.fixed_mixup_alpha_val)
        elif self.hparams.augmentation_type == "none":
            self.augmenter = NoAug()
        else:
            raise ValueError(f"Unknown aug_type: {self.hparams.augmentation_type}")

        # Register normalization constants as buffers (moved to device in forward)
        self.register_buffer('mean_tf', MEAN.view(1,3,1,1))
        self.register_buffer('std_tf', STD.view(1,3,1,1))

    def forward(self, x_input):
        """
        Defines the forward pass of the model.
        Includes normalization before passing to the network.
        """
        # Ensure normalization constants are on the same device as input x_input
        mean_transformed = self.mean_tf.to(x_input.device)
        std_transformed = self.std_tf.to(x_input.device)
        normalized_x = (x_input - mean_transformed) / std_transformed
        return self.net(normalized_x)

    def training_step(self, batch_data, batch_idx):
        input_images, original_labels = batch_data # Data from DataLoader
        batch_size = input_images.size(0)

        # Ensure labels are on the correct device
        original_labels = original_labels.to(self.device)

        # Apply augmentation
        if self.augmenter.requires_labels: # For Mixup-like augmentations
            # Augmenter is an nn.Module, PL moves it to self.device.
            # Input_images are also moved to self.device by PL.
            # Output tensors from augmenter should be on self.device.
            augmented_images, labels_1, labels_2, lambda_coeffs = self.augmenter(input_images, original_labels)
            
            # Ensure all components for loss are on self.device (safeguard)
            labels_1 = labels_1.to(self.device)
            labels_2 = labels_2.to(self.device)
            lambda_coeffs = lambda_coeffs.to(self.device)

            logits_output = self(augmented_images) # Forward pass with augmented images

            # Mixup cross-entropy loss
            loss_ce_1 = F.cross_entropy(logits_output, labels_1, reduction='none')
            loss_ce_2 = F.cross_entropy(logits_output, labels_2, reduction='none')
            cross_entropy_loss = (lambda_coeffs * loss_ce_1 + (1.0 - lambda_coeffs) * loss_ce_2).mean()

            # Accuracy calculation for Mixup (predict based on higher lambda coefficient)
            predictions = logits_output.argmax(dim=1)
            correct_preds_mask_1 = (predictions == labels_1) & (lambda_coeffs >= 0.5)
            correct_preds_mask_2 = (predictions == labels_2) & (lambda_coeffs < 0.5)
            accuracy = (correct_preds_mask_1 | correct_preds_mask_2).float().mean()
        else: # For NoAug or other non-label-mixing augmentations
            augmented_images = self.augmenter(input_images) # NoAug just returns input_images
            logits_output = self(augmented_images)
            cross_entropy_loss = F.cross_entropy(logits_output, original_labels)
            accuracy = (logits_output.argmax(dim=1) == original_labels).float().mean()

        # KL divergence for augmentation parameters (if applicable, e.g., LearnableMixup)
        # self.augmenter.kl() returns a tensor on the augmenter's device (self.device)
        augmenter_kl_loss = self.augmenter.kl()
        
        # Total loss: CE + (beta_aug * KL_aug / N)
        # Weight decay for network parameters is handled by AdamW optimizer directly.
        total_loss = cross_entropy_loss + \
                     self.hparams.beta_augmenter_reg * augmenter_kl_loss / batch_size

        # Logging
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_data = {"train_loss": total_loss, "train_acc": accuracy,
                    "train_ce_loss": cross_entropy_loss, "lr": current_lr}

        if isinstance(self.augmenter, LearnableMixup):
             log_data["mixup_logit_mu"] = self.augmenter.mu.item()
             log_data["mixup_log_sigma"] = self.augmenter.log_sigma.item()
             with torch.no_grad():
                 logit_alpha_dist = torch.distributions.Normal(self.augmenter.mu, torch.exp(self.augmenter.log_sigma))
                 alpha_samples = torch.sigmoid(logit_alpha_dist.sample((1000,))) # Sample to estimate
                 log_data["est_mixup_alpha_mean"] = alpha_samples.mean().item()
                 log_data["est_mixup_alpha_std"] = alpha_samples.std().item()
             log_data["mixup_kl_scaled"] = (self.hparams.beta_augmenter_reg * augmenter_kl_loss / batch_size).item()
             log_data["mixup_kl_raw"] = augmenter_kl_loss.item()


        self.log_dict(log_data, prog_bar=True, on_step=True, on_epoch=False, batch_size=batch_size)
        return total_loss

    def validation_step(self, batch_data, batch_idx):
        with torch.no_grad(): # Disable gradient calculations for validation
            input_images, true_labels_cpu = batch_data # input_images on device, true_labels on CPU
            true_labels = true_labels_cpu.to(self.device) # Move labels to device

            # No augmentation during validation, pass original images
            logits_output = self(input_images) 
            validation_loss = F.cross_entropy(logits_output, true_labels)
            validation_accuracy = (logits_output.argmax(dim=1) == true_labels).float().mean()

        # Log validation metrics (sync across GPUs for DDP)
        self.log_dict({"val_loss": validation_loss, "val_acc": validation_accuracy},
                       prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return {"val_loss": validation_loss, "val_acc": validation_accuracy}

    def configure_optimizers(self):
        # Separate parameters for network and augmenter for potentially different learning rates/decay
        network_params = [p for n, p in self.named_parameters() if n.startswith("net.") and p.requires_grad]
        augmenter_params = [p for n, p in self.named_parameters() if n.startswith("augmenter.") and p.requires_grad]

        param_groups = []
        if not network_params:
            raise ValueError("No network parameters found for the optimizer.")
        param_groups.append({
            "params": network_params, "lr": self.hparams.learning_rate,
            "weight_decay": self.hparams.beta_network_reg # AdamW handles weight decay
        })

        if augmenter_params: # If augmenter has learnable parameters
             param_groups.append({
                 "params": augmenter_params,
                 "lr": self.hparams.learning_rate * 10, # Example: 10x faster LR for augmenter params
                 "weight_decay": 0.0 # Typically no weight decay for augmentation params
            })
        elif self.hparams.augmentation_type == "learnable_mixup" and not augmenter_params:
             # This warning should ideally not trigger if LearnableMixup is used.
             print("Warning: LearnableMixup selected, but no augmenter parameters found for optimizer.")

        optimizer = torch.optim.AdamW(param_groups) # Default LR for AdamW is taken from the first group if not specified per group
        
        # Cosine Annealing scheduler with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,     # Number of epochs for the first restart cycle
            T_mult=2,   # Factor by which T_i is multiplied after a restart (T_i = T_0 * T_mult^i)
            eta_min=self.hparams.learning_rate / 100 # Minimum learning rate
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

# ───────────────────── main routine ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ImageNet Training/Evaluation with Mixup variants")
    parser.add_argument('--epochs',type=int,default=90, help="Total training epochs.")
    parser.add_argument('--batch', type=int,default=256, help="Batch size per GPU.")
    parser.add_argument('--workers',type=int,default=8, help="Dataloader workers.")
    parser.add_argument('--lr',type=float,default=1e-4, help="Network learning rate (AdamW).")
    parser.add_argument('--gpus',type=int,default=-1, help="Num GPUs (-1 for all, 0 for CPU).")
    parser.add_argument('--strategy', type=str, default=None, help="Distributed strategy (e.g., ddp, ddp_find_unused_parameters_false). Auto if None.")
    parser.add_argument('--precision', type=str, default="16", help="Precision: '16', 'bf16', '32', '64'.")
    parser.add_argument('--ckpt_dir_base',default=None, help="Base directory for all checkpoints. Subdirs per aug_type created under it.")
    parser.add_argument('--eval_only',action='store_true', help="Evaluation mode, no training.")
    parser.add_argument('--ckpt_path',default=None, help="Path to checkpoint for eval or resuming.")
    parser.add_argument('--resume', action='store_true', help="Resume training from ckpt_path.")
    parser.add_argument('--aug_type',type=str,default="learnable_mixup", choices=["learnable_mixup", "fixed_mixup", "none"])
    parser.add_argument('--fixed_mixup_alpha',type=float,default=0.2, help="Alpha for fixed_mixup or initial_alpha for learnable.")
    parser.add_argument('--beta_net', type=float, default=0.01, help="Weight decay for network parameters.")
    parser.add_argument('--beta_aug', type=float, default=1.0, help="KL divergence weight for learnable aug parameters.")
    parser.add_argument('--seed', type=int, default=42, help="Global random seed.")
    parser.add_argument('--no_wandb', action='store_true', help="Disable WandB logging.")
    parser.add_argument('--matmul_precision', type=str, default='medium', choices=['medium', 'high', 'highest'], help="torch.set_float32_matmul_precision")
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")

    args = parser.parse_args()

    if torch.cuda.is_available():
         print(f"Setting torch.set_float32_matmul_precision('{args.matmul_precision}')")
         torch.set_float32_matmul_precision(args.matmul_precision)
    pl.seed_everything(args.seed, workers=True)

    # Determine devices and strategy
    current_device_type, num_devices, selected_strategy = "cpu", 1, None
    if args.gpus == 0: pass # Default is CPU, 1 device
    elif torch.cuda.is_available():
        available_gpus = torch.cuda.device_count()
        actual_gpus = available_gpus if args.gpus == -1 else min(args.gpus, available_gpus)
        if actual_gpus > 0:
            current_device_type, num_devices = "gpu", actual_gpus
            selected_strategy = args.strategy if args.strategy else ("ddp" if num_devices > 1 else "auto")
        else: print("Warning: GPUs requested but none available/usable. Defaulting to CPU.")
    else: print("Warning: CUDA not available. Defaulting to CPU.")
    print(f"Using {num_devices} {current_device_type}(s). Strategy: {selected_strategy}")

    # Pass current_device_type to loaders for pin_memory optimization
    train_dataloader, val_dataloader_cfg, val_transforms_obj = loaders(args.batch, args.workers, current_device_type)
    imagenet_c_dataloaders = loaders_imagenet_c(val_transforms_obj, args.batch, args.workers, current_device_type)

    model_hyperparams = {
        "learning_rate": args.lr, "beta_network_reg": args.beta_net,
        "beta_augmenter_reg": args.beta_aug, "augmentation_type": args.aug_type,
        "fixed_mixup_alpha_val": args.fixed_mixup_alpha
    }

    base_ckpt_dir = args.ckpt_dir_base
    if base_ckpt_dir is None:
        base_ckpt_dir = CKPT_DIR_LEARN if args.aug_type == "learnable_mixup" else CKPT_DIR_FIXED
        if args.aug_type == "none": base_ckpt_dir = "./checkpoints_no_aug_eval"
    final_ckpt_dir = os.path.join(base_ckpt_dir, args.aug_type)
    os.makedirs(final_ckpt_dir, exist_ok=True)
    print(f"Checkpoints/Logs targeting directory: {final_ckpt_dir}")

    wandb_logger = None
    if not args.no_wandb:
        run_name_parts = [PROJECT, args.aug_type]
        if args.aug_type != "none": run_name_parts.append(f"alpha{args.fixed_mixup_alpha}")
        run_name_parts.append(f"seed{args.seed}")
        if args.eval_only: run_name_parts.append("eval")
        run_name = "_".join(map(str,run_name_parts)) # Ensure all parts are strings for join
        wandb_logger = WandbLogger(project=PROJECT, name=run_name, save_dir=final_ckpt_dir)
        wandb_logger.log_hyperparams(vars(args))

    trained_model, trainer_for_fit = None, None
    if args.eval_only:
        if args.aug_type == "none":
            trained_model = LitImageNetModel(load_pretrained_for_no_aug_eval=True, **model_hyperparams)
        elif args.ckpt_path:
            if not os.path.exists(args.ckpt_path): raise FileNotFoundError(f"Eval ckpt: {args.ckpt_path}")
            trained_model = LitImageNetModel.load_from_checkpoint(args.ckpt_path, map_location='cpu', **model_hyperparams)
        else: raise ValueError("For eval_only with mixup, --ckpt_path is required.")
    else:
        if args.aug_type == "none": print("Error: Training with 'none' not supported."); return
        
        checkpoint_cb = ModelCheckpoint(dirpath=final_ckpt_dir, monitor="val_acc", mode="max",
                                        filename=f"{args.aug_type}-best-{{epoch:02d}}-{{val_acc:.3f}}",
                                        save_top_k=1, save_last=False)
        lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')
        trainer_callbacks = [checkpoint_cb, lr_monitor_cb]
        
        # Set precision for trainer
        # For PL < 2.0, "16" often implies native AMP. For PL 2.0+, "16-mixed" is explicit.
        # Using args.precision directly as requested.
        trainer_precision = args.precision

        trainer_for_fit = pl.Trainer(
            max_epochs=args.epochs, accelerator=current_device_type, devices=num_devices,
            strategy=selected_strategy, precision=trainer_precision,
            logger=wandb_logger, callbacks=trainer_callbacks, log_every_n_steps=50,
            gradient_clip_val=args.grad_clip_val if args.grad_clip_val > 0 else None # Apply grad clipping
        )

        ckpt_to_resume = args.ckpt_path if args.resume else None
        if ckpt_to_resume and not os.path.exists(ckpt_to_resume):
            raise FileNotFoundError(f"Resume ckpt: {ckpt_to_resume}")

        # When resuming, PL loads hparams from ckpt. Pass **model_hyperparams to allow CLI to override.
        model_for_training = LitImageNetModel(**model_hyperparams) if not ckpt_to_resume else \
                             LitImageNetModel.load_from_checkpoint(ckpt_to_resume, **model_hyperparams)
        if ckpt_to_resume: print(f"Resuming from {ckpt_to_resume}.")
        else: print("Starting new training run.")

        if wandb_logger: wandb_logger.watch(model_for_training, log='all', log_freq=100)
        
        trainer_for_fit.fit(model_for_training, train_dataloader, val_dataloader_cfg, ckpt_path=ckpt_to_resume)

        best_model_path = checkpoint_cb.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Training finished. Loading best model from: {best_model_path}")
            trained_model = LitImageNetModel.load_from_checkpoint(best_model_path, map_location='cpu')
        else:
            print("Warning: No best model checkpoint saved. Using model state at end of training.")
            trained_model = model_for_training

    if trained_model is None: print("No model for evaluation. Exiting."); return

    # Final evaluation on a single device for precise metrics
    final_eval_accelerator, final_eval_devices, final_eval_strategy = current_device_type, 1, None
    # Default to 32-bit for final eval if training was low precision, unless 32/64 was specified
    eval_precision = args.precision if args.precision in ["32", "64", "32-true", "64-true"] else "32" 
    
    print(f"\n--- Creating final evaluation trainer with {final_eval_devices} {final_eval_accelerator} device(s), precision {eval_precision} ---")
    eval_trainer = pl.Trainer(accelerator=final_eval_accelerator, devices=final_eval_devices,
                              precision=eval_precision, logger=False, strategy=final_eval_strategy)

    print(f"\n--- Validating Clean ({trained_model.hparams.augmentation_type}) ---")
    val_results = eval_trainer.validate(trained_model, val_dataloader_cfg, verbose=False)
    clean_acc = val_results[0]['val_acc'] if val_results else 0.0
    clean_loss = val_results[0]['val_loss'] if val_results else float('nan')
    print(f"Clean ImageNet Acc@1: {clean_acc*100:5.2f}%, Loss: {clean_loss:.4f}")
    if wandb_logger: wandb_logger.log_metrics({"final_clean_acc": clean_acc, "final_clean_loss": clean_loss})

    print(f"\n--- Evaluating ImageNet-C ({trained_model.hparams.augmentation_type}) ---")
    if not imagenet_c_dataloaders: print("Skipping ImageNet-C: no dataloaders.")
    else:
        c_errors_sev, c_mean_err_corr, all_c_accs = {}, {}, []
        for name, sev, loader_c in imagenet_c_dataloaders:
            print(f"Eval: {name} s{sev}...")
            res_c = eval_trainer.validate(trained_model, loader_c, verbose=False)
            acc_c = res_c[0]['val_acc'] if res_c else float('nan')
            err_c = 1.0 - acc_c if not math.isnan(acc_c) else float('nan')
            if not math.isnan(acc_c): all_c_accs.append(acc_c)
            if name not in c_errors_sev: c_errors_sev[name] = [float('nan')] * 5
            if 1<=sev<=5: c_errors_sev[name][sev-1]=err_c; print(f"{name:20s} s{sev}: Acc {acc_c*100:5.2f}% | Err {err_c*100:5.2f}%")
        
        list_corr_avg_errs = []
        for corr, err_list in sorted(c_errors_sev.items()):
            valid_errs = [e for e in err_list if not math.isnan(e)]
            if len(valid_errs)==5: avg_err=np.mean(valid_errs);c_mean_err_corr[corr]=avg_err;list_corr_avg_errs.append(avg_err);print(f"{corr:20s}:AvgErr {avg_err*100:5.2f}%")
            else: c_mean_err_corr[corr]=float('nan')
        mCE = np.mean(list_corr_avg_errs) if list_corr_avg_errs else float('nan')
        mean_c_acc = np.mean(all_c_accs) if all_c_accs else float('nan')
        print(f"\nImNet-C mCE:{mCE*100:5.2f}%"); print(f"ImNet-C MeanAcc:{mean_c_acc*100:5.2f}%")
        if wandb_logger: wandb_logger.log_metrics({"final_imnet_c_mCE": mCE, "final_imnet_c_mean_acc": mean_c_acc})
        
        json_hparams = dict(trained_model.hparams) # Get hparams from the actual model used for eval
        # Ensure CLI args that might not be in model.hparams (if loaded from old ckpt) are included
        for k_arg, v_arg in vars(args).items():
            if k_arg not in json_hparams: json_hparams[k_arg] = v_arg

        output_data = {"cli_args":vars(args),"model_hparams":json_hparams,"clean_accuracy":clean_acc,"clean_loss":clean_loss,
                       "imagenet_c_errors_per_severity":c_errors_sev,"imagenet_c_mean_error_per_corruption":c_mean_err_corr,
                       "mCE_unnormalized":mCE,"mean_accuracy_on_corrupted":mean_c_acc}
        json_parts=['results',trained_model.hparams.augmentation_type,f'alpha{trained_model.hparams.fixed_mixup_alpha_val}',f'seed{args.seed}']
        json_fname="_".join(map(str,json_parts))+".json";json_fpath=os.path.join(final_ckpt_dir,json_fname)
        try:
            with open(json_fpath,'w')as f:
                def np_encoder(o): return o.item()if isinstance(o,(np.generic,np.ndarray))and o.size==1 else(o.tolist()if isinstance(o,(np.ndarray,torch.Tensor))else str(o))
                json.dump(output_data,f,indent=2,default=np_encoder)
            print(f"Results saved to {json_fpath}")
        except Exception as e:print(f"JSON save error:{e}")

    print("\nRun complete.")
    if wandb_logger and wandb_logger.experiment:
        if trainer_for_fit and hasattr(trainer_for_fit,'global_rank')and trainer_for_fit.global_rank!=0:
            if torch.distributed.is_available()and torch.distributed.is_initialized():torch.distributed.barrier()
        if not trainer_for_fit or not(hasattr(trainer_for_fit,'global_rank')and trainer_for_fit.global_rank!=0):
            wandb_logger.experiment.finish()

if __name__=="__main__":
    main()