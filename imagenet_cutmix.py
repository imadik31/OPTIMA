# ------------------------------------------------------------
#  ImageNet training with *learnable* CutMix alpha, fixed CutMix,
#  or no augmentation (eval only).
#  Lightning-based; comes with ImageNet-C evaluation.
#  Corrected rand_bbox for GPU tensors.
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
import wandb # Explicit import
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

PROJECT        = "LearnableCutMix_IMNET_Full" # WandB Project Name
CKPT_DIR_CUTMIX_LEARN = "./checkpoints_learnable_cutmix" # Default for learnable CutMix
CKPT_DIR_CUTMIX_FIXED = "./checkpoints_fixed_cutmix"     # Default for fixed CutMix

# Standard ImageNet normalization constants
MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float)
STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float)

# List of corruption types in ImageNet-C
CORRUPTIONS = [
 "gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur",
 "motion_blur","zoom_blur","snow","frost","fog","brightness","contrast",
 "elastic_transform","pixelate","jpeg_compression"
]

# ───────────────────────── CutMix Helper ───────────────────────
def rand_bbox(size, lam): # lam is patch_area_ratio here
    """
    Generates a random bounding box for CutMix.
    Args:
        size: Size of the image, e.g., (C, H, W) for a single image.
        lam: Area ratio of the patch to be cut from the *second* image.
             This is typically (1 - lambda_from_beta_for_first_image).
             Expected to be a Python float or a 0-dim tensor.
    Returns:
        (bbx1, bby1, bbx2, bby2): Bounding box coordinates.
    """
    # Determine image width (W) and height (H) from the input size
    if len(size) == 4: W, H = size[2], size[3] # BxCxHxW (less common here)
    elif len(size) == 3: W, H = size[1], size[2] # CxHxW (expected from augmenter loop)
    else: raise ValueError(f"Unsupported size format for rand_bbox: {size}")

    # Convert lam to a CPU float if it's a tensor (e.g., from GPU)
    lam_float = lam.item() if isinstance(lam, torch.Tensor) else float(lam)
    # Ensure lam_float (patch area ratio) is within [0, 1]
    lam_float = np.clip(lam_float, 0.0, 1.0)

    # Handle edge cases for lam_float
    if lam_float == 0.0: return 0,0,0,0 # No patch to cut
    if lam_float == 1.0: return 0,0,W,H # Patch covers the entire image area

    # Calculate patch dimensions based on lam_float (area ratio)
    cut_rat = np.sqrt(lam_float) # Ratio of patch side to image side
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)

    # If image or calculated patch dimensions are zero, no valid patch can be made
    if W == 0 or H == 0 or cut_w == 0 or cut_h == 0 : return 0,0,0,0
    
    # Uniformly random center (cx, cy) for the patch
    # Ensure cx, cy are within valid image bounds to pick a center
    # (np.random.randint upper bound is exclusive)
    cx, cy = np.random.randint(W), np.random.randint(H)
    
    # Calculate bounding box coordinates, clamping to image boundaries
    bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
    bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
    
    # Ensure the box has a valid positive area after clipping
    if bbx1 >= bbx2 or bby1 >= bby2:
        return 0,0,0,0 # Return zero-area box if clipping results in invalid dimensions

    return bbx1, bby1, bbx2, bby2

# ───────────────────────── data loaders ──────────────────────
def loaders(batch_size_arg, num_workers_arg, current_device_type="cpu"):
    """
    Creates DataLoaders for ImageNet training and validation.
    Applies standard ToTensor and augmentations.
    """
    train_transforms = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()])
    val_transforms = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    if not os.path.isdir(IMAGENET_TRAIN): raise FileNotFoundError(f"Train dir: {IMAGENET_TRAIN}")
    if not os.path.isdir(IMAGENET_VAL): raise FileNotFoundError(f"Val dir: {IMAGENET_VAL}")
    train_ds = dset.ImageFolder(IMAGENET_TRAIN, train_transforms)
    val_ds = dset.ImageFolder(IMAGENET_VAL, val_transforms)
    use_persistent = num_workers_arg > 0
    pin_memory_flag = (current_device_type == "gpu")
    train_loader = DataLoader(train_ds, batch_size=batch_size_arg, shuffle=True, num_workers=num_workers_arg, pin_memory=pin_memory_flag, persistent_workers=use_persistent)
    val_loader = DataLoader(val_ds, batch_size=batch_size_arg, shuffle=False, num_workers=num_workers_arg, pin_memory=pin_memory_flag, persistent_workers=use_persistent)
    return train_loader, val_loader, val_transforms

def loaders_imagenet_c(transforms_arg, batch_size_arg, num_workers_arg, current_device_type="cpu"):
    """
    Creates DataLoaders for ImageNet-C, using specified transforms.
    """
    dataloaders_c = []
    if not os.path.isdir(IMAGENET_C): print(f"Warning: ImageNet-C dir not found: {IMAGENET_C}"); return []
    use_persistent = num_workers_arg > 0
    pin_memory_flag = (current_device_type == "gpu")
    for corr in CORRUPTIONS:
        corr_path = os.path.join(IMAGENET_C, corr);
        if not os.path.isdir(corr_path): continue
        for sev in range(1,6):
            sev_path = os.path.join(corr_path, str(sev));
            if not os.path.isdir(sev_path): continue
            ds_c = dset.ImageFolder(sev_path, transforms_arg)
            loader_c = DataLoader(ds_c, batch_size=batch_size_arg, shuffle=False, num_workers=num_workers_arg, pin_memory=pin_memory_flag, persistent_workers=use_persistent)
            dataloaders_c.append((corr, sev, loader_c))
    if not dataloaders_c: print(f"Warning: No ImageNet-C data loaded from {IMAGENET_C}.")
    return dataloaders_c

# ─────────────────── Augmenter Base Class ───────────────────
class Augmenter(nn.Module):
    """Base class for augmenters. Defines interface."""
    requires_labels = False
    def __init__(self): super().__init__()
    def forward(self, x_input, y_input=None): raise NotImplementedError
    def kl(self): # Default KL is 0
        device = 'cpu'
        if hasattr(self, 'log_beta_alpha_mu'): device = self.log_beta_alpha_mu.device
        elif hasattr(self, 'mu'): device = self.mu.device
        elif hasattr(self, 'dummy_param'): device = self.dummy_param.device
        else:
            try: device = next(self.parameters()).device
            except StopIteration:
                try: device = next(self.buffers()).device
                except StopIteration: pass
        return torch.tensor(0.0, device=device)

# ─────────────────── No Augmentation Wrapper ──────────────────
class NoAug(Augmenter):
    """Identity augmenter."""
    def __init__(self): super().__init__(); self.register_buffer("dummy_param", torch.empty(0))
    def forward(self, x_input, y_input=None): return x_input

# ─────────────────── Fixed CutMix augmenter ──────────────────
class FixedCutMix(Augmenter):
    """
    Implements CutMix augmentation with a fixed alpha for the Beta distribution.
    lambda_prop_img1 ~ Beta(alpha, alpha)
    A patch from a second image (area ratio 1 - lambda_prop_img1) is pasted onto the first.
    Labels are mixed according to the final area proportions.
    """
    requires_labels = True
    def __init__(self, beta_alpha_val=1.0): # Default CutMix alpha for Beta is 1.0
        super().__init__()
        if beta_alpha_val <= 0: raise ValueError("Beta alpha for CutMix > 0.")
        self.beta_alpha = beta_alpha_val
        self.register_buffer("dummy_param", torch.empty(0)) # For KL device detection

    def forward(self, x_input, y_input): # x_input: (B,C,H,W), y_input: (B,)
        B, C, H, W = x_input.size()
        dev = x_input.device # Data is already on target device from DataLoader/PL

        beta_param = torch.tensor(self.beta_alpha, dtype=torch.float, device=dev)
        # lambda_prop_img1: Proportion of the first image/label, sampled from Beta(alpha,alpha)
        lambda_prop_img1 = torch.distributions.Beta(beta_param, beta_param).sample((B,)).to(dev)
        # Clamp for numerical stability if used in loss calculations needing strict (0,1)
        lambda_prop_img1 = torch.clamp(lambda_prop_img1, 1e-6, 1.0 - 1e-6)

        # Get shuffled batch for mixing
        perm_idx = torch.randperm(B, device=dev)
        x_shuffled, y_shuffled = x_input[perm_idx], y_input[perm_idx]
        
        x_mixed = x_input.clone()
        # final_lambda_img1: Actual coefficient for y_input after patch application
        final_lambda_img1 = torch.ones_like(lambda_prop_img1) # Initialize assuming no patch

        for i in range(B):
            # patch_area_ratio: Desired area of the patch from x_shuffled[i]
            patch_area_ratio = 1.0 - lambda_prop_img1[i] # This value is passed to rand_bbox as 'lam'
            
            # Get bounding box for the patch. x_input[i].size() is (C,H,W)
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_input[i].size(), patch_area_ratio)
            
            # If a valid patch box is generated
            if (bbx2 - bbx1) > 0 and (bby2 - bby1) > 0:
                # Paste patch from shuffled image into the mixed image
                x_mixed[i, :, bby1:bby2, bbx1:bbx2] = x_shuffled[i, :, bby1:bby2, bbx1:bbx2]
                # Calculate actual proportion of image1 remaining
                actual_patch_area_prop = (bbx2 - bbx1) * (bby2 - bby1) / (H * W)
                final_lambda_img1[i] = 1.0 - actual_patch_area_prop
            else: # No patch applied (e.g., patch_area_ratio was 0, or rand_bbox returned 0-area)
                final_lambda_img1[i] = 1.0 # Original image fully kept

        final_lambda_img1 = torch.clamp(final_lambda_img1, 0.0, 1.0) # Ensure valid range [0,1]
        return x_mixed, y_input, y_shuffled, final_lambda_img1

# ─────────────────── learnable CutMix augmenter ───────────────
class LearnableCutMix(Augmenter):
    """
    Implements CutMix where alpha for Beta(alpha,alpha) is learned.
    Learns mu and log_sigma for Normal dist over log(alpha_for_beta).
    alpha_for_beta_i ~ exp(Normal(mu, sigma^2))
    lambda_prop_img1_i ~ Beta(alpha_for_beta_i, alpha_for_beta_i)
    """
    requires_labels = True
    def __init__(self, initial_beta_alpha=1.0, prior_standard_dev=2.0, prior_mean_offset_val=0.0):
        super().__init__()
        # Clamp initial alpha for log and stability
        init_beta_alpha_clamped = np.clip(initial_beta_alpha, 1e-4, 100.0) # Keep alpha positive and reasonable
        
        # Learnable parameters for the Normal distribution of log(alpha_for_beta)
        self.log_beta_alpha_mu = nn.Parameter(torch.log(torch.tensor(init_beta_alpha_clamped, dtype=torch.float)))
        self.log_beta_alpha_log_sigma = nn.Parameter(torch.log(torch.tensor(0.1, dtype=torch.float))) # Small initial variance

        # Fixed prior distribution for log(alpha_for_beta)
        prior_mu = torch.log(torch.tensor(init_beta_alpha_clamped, dtype=torch.float)) + prior_mean_offset_val
        self.register_buffer("prior_log_beta_alpha_mu_buffer", prior_mu)
        self.register_buffer("prior_log_beta_alpha_std_buffer", torch.tensor(prior_standard_dev, dtype=torch.float))

    def forward(self, x_input, y_input):
        B, C, H, W = x_input.size()
        dev = self.log_beta_alpha_mu.device # Parameters are on correct device

        # Sample log(alpha_for_beta_i) per image in batch
        eps = torch.randn(B, device=dev) # (B,)
        # log_beta_alpha_mu and log_beta_alpha_log_sigma are scalar params, broadcast with eps
        log_beta_alpha_samples = self.log_beta_alpha_mu + torch.exp(self.log_beta_alpha_log_sigma) * eps
        
        # Transform to alpha_for_beta > 0
        beta_alpha_params = torch.exp(log_beta_alpha_samples)
        beta_alpha_params_clamped = torch.clamp(beta_alpha_params, 1e-4, 100.0) # Clamp for Beta stability

        # Sample lambda_prop_img1_i ~ Beta(alpha_for_beta_i, alpha_for_beta_i)
        lambda_prop_img1 = torch.distributions.Beta(beta_alpha_params_clamped, beta_alpha_params_clamped).sample() # Already on dev
        lambda_prop_img1 = torch.clamp(lambda_prop_img1, 1e-6, 1.0 - 1e-6)

        perm_idx = torch.randperm(B, device=dev)
        x_shuffled, y_shuffled = x_input[perm_idx], y_input[perm_idx]
        x_mixed = x_input.clone()
        final_lambda_img1 = torch.ones_like(lambda_prop_img1)

        for i in range(B):
            patch_area_ratio = 1.0 - lambda_prop_img1[i]
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_input[i].size(), patch_area_ratio)
            if (bbx2 - bbx1) > 0 and (bby2 - bby1) > 0:
                x_mixed[i, :, bby1:bby2, bbx1:bbx2] = x_shuffled[i, :, bby1:bby2, bbx1:bbx2]
                actual_patch_area_prop = (bbx2 - bbx1) * (bby2 - bby1) / (H * W)
                final_lambda_img1[i] = 1.0 - actual_patch_area_prop
            else:
                final_lambda_img1[i] = 1.0
        final_lambda_img1 = torch.clamp(final_lambda_img1, 0.0, 1.0)
        return x_mixed, y_input, y_shuffled, final_lambda_img1

    def kl(self):
        """KL( N(learned_mu, learned_sigma^2) || N(prior_mu, prior_sigma^2) ) for log(alpha_for_beta)"""
        cur_sig_sq = torch.exp(self.log_beta_alpha_log_sigma * 2) # learned_sigma^2
        prior_sig_sq = self.prior_log_beta_alpha_std_buffer**2
        t1 = (cur_sig_sq + (self.log_beta_alpha_mu - self.prior_log_beta_alpha_mu_buffer)**2) / (2 * prior_sig_sq)
        t2 = torch.log(self.prior_log_beta_alpha_std_buffer / torch.exp(self.log_beta_alpha_log_sigma))
        return t1 + t2 - 0.5

# ───────────────────── Lightning module ──────────────────────
class LitImageNetModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, beta_network_reg=0.01, beta_augmenter_reg=1.0,
                 augmentation_type="learnable_cutmix", fixed_cutmix_alpha_val=1.0,
                 load_pretrained_for_no_aug_eval=False):
        super().__init__()
        self.save_hyperparameters()
        net_weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.net = models.resnet50(weights=net_weights)
        if not self.hparams.load_pretrained_for_no_aug_eval: # Modify FC for fine-tuning
            self.net.fc = nn.Linear(self.net.fc.in_features, 1000)

        aug_type = self.hparams.augmentation_type
        if aug_type == "learnable_cutmix": self.augmenter = LearnableCutMix(self.hparams.fixed_cutmix_alpha_val)
        elif aug_type == "fixed_cutmix": self.augmenter = FixedCutMix(self.hparams.fixed_cutmix_alpha_val)
        elif aug_type == "none": self.augmenter = NoAug()
        else: raise ValueError(f"Unknown aug_type: {aug_type}")
        
        self.register_buffer('mean_tf',MEAN.view(1,3,1,1)) # For normalization
        self.register_buffer('std_tf',STD.view(1,3,1,1))  # For normalization

    def forward(self, x): # Input x is assumed to be on self.device
        # Move normalization constants to input's device before operation
        mean_tf_dev = self.mean_tf.to(x.device); std_tf_dev = self.std_tf.to(x.device)
        normalized_x = (x - mean_tf_dev) / std_tf_dev
        return self.net(normalized_x)

    def training_step(self, batch_data, batch_idx):
        input_images, original_labels_cpu = batch_data # input_images from DL is on device, labels may be CPU
        B = input_images.size(0)
        original_labels = original_labels_cpu.to(self.device) # Ensure labels are on model's device

        if self.augmenter.requires_labels: # For CutMix variants
            # self.augmenter is an nn.Module, its params are on self.device.
            # It operates on input_images (on self.device) and should return tensors on self.device.
            augmented_images, labels_1, labels_2, final_lambda_img1 = self.augmenter(input_images, original_labels)
            
            # Safeguard: ensure all returned components for loss are on the correct device.
            # labels_1 should be original_labels (already moved). labels_2 from permuted y_input.
            # final_lambda_img1 is created with .ones_like which matches device, then ops preserve it.
            labels_2 = labels_2.to(self.device) 
            final_lambda_img1 = final_lambda_img1.to(self.device)

            logits_output = self(augmented_images)
            loss_ce_1 = F.cross_entropy(logits_output, labels_1, reduction='none')
            loss_ce_2 = F.cross_entropy(logits_output, labels_2, reduction='none')
            # final_lambda_img1 is the coefficient for labels_1
            cross_entropy_loss = (final_lambda_img1 * loss_ce_1 + (1.0 - final_lambda_img1) * loss_ce_2).mean()
            preds = logits_output.argmax(dim=1)
            acc = ((preds == labels_1)&(final_lambda_img1 >= 0.5) | (preds == labels_2)&(final_lambda_img1 < 0.5)).float().mean()
        else: # For NoAug
            augmented_images = self.augmenter(input_images) # NoAug is identity
            logits_output = self(augmented_images)
            cross_entropy_loss = F.cross_entropy(logits_output, original_labels)
            acc = (logits_output.argmax(dim=1) == original_labels).float().mean()

        aug_kl_loss = self.augmenter.kl() # Returns KL tensor on self.device
        # Total loss = CE + (beta_aug * KL_aug / N_batch)
        # Network L2/weight decay is handled by AdamW optimizer.
        total_loss = cross_entropy_loss + self.hparams.beta_augmenter_reg * aug_kl_loss / B

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        log_data = {"train_loss":total_loss,"train_acc":acc,"train_ce_loss":cross_entropy_loss,"lr":lr}
        if isinstance(self.augmenter, LearnableCutMix):
            log_data["cutmix_log_beta_alpha_mu"] = self.augmenter.log_beta_alpha_mu.item()
            log_data["cutmix_log_beta_alpha_log_sigma"] = self.augmenter.log_beta_alpha_log_sigma.item()
            log_data["est_cutmix_beta_alpha"] = torch.exp(self.augmenter.log_beta_alpha_mu).item()
            log_data["cutmix_kl_scaled"] = (self.hparams.beta_augmenter_reg * aug_kl_loss / B).item()
            log_data["cutmix_kl_raw"] = aug_kl_loss.item() # Log raw KL
        self.log_dict(log_data, prog_bar=True, on_step=True, on_epoch=False, batch_size=B)
        return total_loss

    def validation_step(self, batch_data, batch_idx):
        with torch.no_grad(): # Disable gradients for validation
            input_images, true_labels_cpu = batch_data # input_images on device by PL
            true_labels = true_labels_cpu.to(self.device) # Move labels to model's device
            logits_output = self(input_images) # No augmentation in validation
            val_loss = F.cross_entropy(logits_output, true_labels)
            val_acc = (logits_output.argmax(dim=1) == true_labels).float().mean()
        self.log_dict({"val_loss":val_loss,"val_acc":val_acc},prog_bar=True,on_epoch=True,sync_dist=True)
        return {"val_loss":val_loss,"val_acc":val_acc}

    def configure_optimizers(self):
        net_p=[p for n,p in self.named_parameters() if n.startswith("net.")and p.requires_grad]
        aug_p=[p for n,p in self.named_parameters() if n.startswith("augmenter.")and p.requires_grad]
        groups=[]
        if not net_p: raise ValueError("No net params.")
        groups.append({"params":net_p,"lr":self.hparams.learning_rate,"weight_decay":self.hparams.beta_network_reg})
        if aug_p: groups.append({"params":aug_p,"lr":self.hparams.learning_rate*10,"weight_decay":0.0})
        elif self.hparams.augmentation_type=="learnable_cutmix" and not aug_p:
            # This should ideally not happen if LearnableCutMix is correctly initialized with nn.Parameters
             print("Warning: LearnableCutMix but no aug params found for optim. This is unexpected.")
        opt=torch.optim.AdamW(groups) # AdamW handles weight decay internally for groups with weight_decay > 0
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=15,T_mult=2,eta_min=self.hparams.learning_rate/100)
        return {"optimizer":opt,"lr_scheduler":{"scheduler":sch,"interval":"epoch"}}

# ───────────────────── main routine ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ImageNet Training with CutMix variants")
    parser.add_argument('--epochs',type=int,default=90);parser.add_argument('--batch', type=int,default=256)
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--gpus',type=int,default=-1);parser.add_argument('--strategy',type=str,default=None)
    parser.add_argument('--precision',type=str,default="16",help="Simple: '16','bf16','32','64'.")
    parser.add_argument('--ckpt_dir_base',default=None);parser.add_argument('--eval_only',action='store_true')
    parser.add_argument('--ckpt_path',default=None);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--aug_type',type=str,default="learnable_cutmix",choices=["learnable_cutmix","fixed_cutmix","none"])
    parser.add_argument('--fixed_cutmix_alpha',type=float,default=1.0, help="Alpha for Beta dist in CutMix (fixed or initial for learnable).")
    parser.add_argument('--beta_net',type=float,default=0.01, help="Weight decay for network parameters.")
    parser.add_argument('--beta_aug',type=float,default=1.0, help="KL divergence weight for learnable CutMix alpha.")
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--no_wandb',action='store_true')
    parser.add_argument('--matmul_precision',type=str,default='medium', choices=['medium','high','highest'])
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    args = parser.parse_args()

    if torch.cuda.is_available():torch.set_float32_matmul_precision(args.matmul_precision)
    pl.seed_everything(args.seed,workers=True)

    dev_t,num_dev,strat="cpu",1,None
    if args.gpus==0:pass
    elif torch.cuda.is_available():
        avail=torch.cuda.device_count();actual=avail if args.gpus==-1 else min(args.gpus,avail)
        if actual>0:dev_t,num_dev,strat="gpu",actual,args.strategy if args.strategy else("ddp"if actual>1 else"auto")
        else:print("Warn: GPUs req but none avail. CPU.")
    else:print("Warn: CUDA not avail. CPU.")
    print(f"Using {num_dev} {dev_t}(s). Strategy: {strat}")

    train_loader,val_loader_cfg,val_tf=loaders(args.batch,args.workers, dev_t) # Pass dev_t for pin_memory
    imnet_c_loaders=loaders_imagenet_c(val_tf,args.batch,args.workers, dev_t)
    
    model_hparams={"learning_rate":args.lr,"beta_network_reg":args.beta_net,"augmentation_type":args.aug_type,
                     "beta_augmenter_reg":args.beta_aug, "fixed_cutmix_alpha_val":args.fixed_cutmix_alpha}

    ckpt_b=args.ckpt_dir_base
    if ckpt_b is None:
        ckpt_b = CKPT_DIR_CUTMIX_LEARN if args.aug_type == "learnable_cutmix" else CKPT_DIR_CUTMIX_FIXED
        if args.aug_type=="none": ckpt_b="./checkpoints_no_aug_eval"
    final_ckpt_dir=os.path.join(ckpt_b,args.aug_type);os.makedirs(final_ckpt_dir,exist_ok=True)
    print(f"Ckpts/logs dir: {final_ckpt_dir}")

    wandb_log=None
    if not args.no_wandb:
        parts=[PROJECT,args.aug_type,f"alpha{args.fixed_cutmix_alpha}",f"kl{args.beta_aug}",f"s{args.seed}"]
        if args.eval_only:parts.append("eval")
        name="_".join(map(str,parts));wandb_log=WandbLogger(project=PROJECT,name=name,save_dir=final_ckpt_dir)
        wandb_log.log_hyperparams(vars(args))

    model,trainer_fit=None,None
    if args.eval_only:
        if args.aug_type=="none":model=LitImageNetModel(load_pretrained_for_no_aug_eval=True,**model_hparams)
        elif args.ckpt_path:
            if not os.path.exists(args.ckpt_path):raise FileNotFoundError(f"Eval ckpt:{args.ckpt_path}")
            model=LitImageNetModel.load_from_checkpoint(args.ckpt_path,map_location='cpu',**model_hparams)
        else:raise ValueError("Eval needs --ckpt_path or aug_type 'none'.")
    else:
        if args.aug_type=="none":print("Err:No training for 'none'.");return
        ckpt_cb=ModelCheckpoint(dirpath=final_ckpt_dir,monitor="val_acc",mode="max",filename=f"{args.aug_type}-best-{{epoch:02d}}-{{val_acc:.3f}}",save_top_k=1,save_last=False)
        lr_cb=LearningRateMonitor(logging_interval='epoch')
        
        trainer_precision = args.precision # Use user-provided string directly

        trainer_fit=pl.Trainer(max_epochs=args.epochs,accelerator=dev_t,devices=num_dev,strategy=strat,
                                 precision=trainer_precision,logger=wandb_log,callbacks=[ckpt_cb,lr_cb],
                                 log_every_n_steps=50, gradient_clip_val=args.grad_clip_val if args.grad_clip_val > 0 else None)
        ckpt_res=args.ckpt_path if args.resume else None
        if ckpt_res and not os.path.exists(ckpt_res):raise FileNotFoundError(f"Resume ckpt:{ckpt_res}")
        
        # When resuming, PL loads hparams from ckpt. Pass **model_hparams to allow CLI to override.
        m_train=LitImageNetModel(**model_hparams)if not ckpt_res else LitImageNetModel.load_from_checkpoint(ckpt_res,**model_hparams)
        if ckpt_res:print(f"Resuming from {ckpt_res}.")
        else:print("New training.")
        if wandb_log:wandb_log.watch(m_train,log='all',log_freq=100)
        trainer_fit.fit(m_train,train_loader,val_loader_cfg,ckpt_path=ckpt_res)
        best_p=ckpt_cb.best_model_path
        if best_p and os.path.exists(best_p):model=LitImageNetModel.load_from_checkpoint(best_p,map_location='cpu')
        else:model=m_train

    if model is None:print("No model.Exit.");return
    
    final_eval_accel,final_eval_devs,final_eval_strat=dev_t,1,None
    eval_prec=args.precision if args.precision in["32","64","32-true","64-true"]else"32"
    eval_trainer=pl.Trainer(accelerator=final_eval_accel,devices=final_eval_devs,precision=eval_prec,logger=False,strategy=final_eval_strat)
    print(f"\n--- Validating Clean ({model.hparams.augmentation_type}) p={eval_prec} ---")
    val_r=eval_trainer.validate(model,val_loader_cfg,verbose=False)
    clean_acc,clean_loss=(val_r[0]['val_acc'],val_r[0]['val_loss'])if val_r else(0.0,float('nan'))
    print(f"Clean Acc@1:{clean_acc*100:5.2f}%,Loss:{clean_loss:.4f}")
    if wandb_log:wandb_log.log_metrics({"final_clean_acc":clean_acc,"final_clean_loss":clean_loss})

    print(f"\n--- Eval ImageNet-C ({model.hparams.augmentation_type}) p={eval_prec} ---")
    if not imnet_c_loaders:print("Skip ImageNet-C:no loaders.")
    else:
        c_err_s,c_mean_e,all_c_a={},{},[]
        for n,s,l_c in imnet_c_loaders:
            print(f"Eval:{n} s{s}...");r_c=eval_trainer.validate(model,l_c,verbose=False)
            a_c=r_c[0]['val_acc']if r_c else float('nan');e_c=1.0-a_c if not math.isnan(a_c)else float('nan')
            if not math.isnan(a_c):all_c_a.append(a_c)
            if n not in c_err_s:c_err_s[n]=[float('nan')]*5
            if 1<=s<=5:c_err_s[n][s-1]=e_c;print(f"{n:20s} s{s}: Acc {a_c*100:5.2f}% | Err {e_c*100:5.2f}%")
        l_corr_avg_e=[]
        for c,e_l in sorted(c_err_s.items()):
            v_e=[e for e in e_l if not math.isnan(e)]
            if len(v_e)==5:avg_e=np.mean(v_e);c_mean_e[c]=avg_e;l_corr_avg_e.append(avg_e);print(f"{c:20s}:AvgErr {avg_e*100:5.2f}%")
            else:c_mean_e[c]=float('nan')
        mCE=np.mean(l_corr_avg_e)if l_corr_avg_e else float('nan')
        m_c_a=np.mean(all_c_a)if all_c_a else float('nan')
        print(f"\nImNet-C mCE:{mCE*100:5.2f}%");print(f"ImNet-C MeanAcc:{m_c_a*100:5.2f}%")
        if wandb_log:wandb_log.log_metrics({"final_imnet_c_mCE":mCE,"final_imnet_c_mean_acc":m_c_a})
        
        js_h=dict(model.hparams)
        for k,v in vars(args).items(): # Ensure CLI args are captured in saved hparams for JSON
            if k.startswith("fixed_cutmix_")or k=="beta_augmenter_reg" or k=="beta_net":js_h[k]=js_h.get(k,v)
        js_o={"cli_args":vars(args),"model_hparams":js_h,"clean_accuracy":clean_acc,"clean_loss":clean_loss,
                  "imagenet_c_errors_per_severity":c_err_s,"imagenet_c_mean_error_per_corruption":c_mean_e,
                  "mCE_unnormalized":mCE,"mean_accuracy_on_corrupted":m_c_a}
        json_fparts=['results',model.hparams.augmentation_type,f'alpha{model.hparams.fixed_cutmix_alpha_val}',f'kl{model.hparams.beta_augmenter_reg}',f'seed{args.seed}']
        json_fname="_".join(map(str,json_fparts))+".json";json_fpath=os.path.join(final_ckpt_dir,json_fname)
        try:
            with open(json_fpath,'w')as f:
                def np_e(o):return o.item()if isinstance(o,(np.generic,np.ndarray))and o.size==1 else(o.tolist()if isinstance(o,(np.ndarray,torch.Tensor))else str(o))
                json.dump(js_o,f,indent=2,default=np_e)
            print(f"Results saved to {json_fpath}")
        except Exception as e:print(f"JSON save error:{e}")

    print("\nRun complete.")
    if wandb_log and wandb_log.experiment:
        # Graceful WandB finish, especially for DDP
        if trainer_fit and hasattr(trainer_fit,'global_rank')and trainer_fit.global_rank!=0:
            if torch.distributed.is_available()and torch.distributed.is_initialized():torch.distributed.barrier()
        # Only rank 0 (or single process) closes the run
        if not trainer_fit or not(hasattr(trainer_fit,'global_rank')and trainer_fit.global_rank!=0):
            wandb_log.experiment.finish()

if __name__=="__main__":
    main()