#!/usr/bin/env python3
# learnable_augmix_jsd_v_type1_imagenet.py
# ------------------------------------------------------------
#  ImageNet training with AugMix where 'aug_severity' is learnable (Type 1),
#  plus JSD loss. Augmenter is an nn.Module.
#  This version implements the "Type 1 Learnable" approach for AugMix severity,
#  where the severity parameter's distribution is learned.
#  PIL-based augmentations are performed within the nn.Module augmenter,
#  which may lead to performance bottlenecks.
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
import wandb # Explicit import for safety
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF # For potential future tensor ops
import torchvision.datasets as dset
import torchvision.models as models
from PIL import Image, ImageOps, ImageEnhance

# ───────────────────────── paths & defaults ──────────────────
# !!! USERS SHOULD UPDATE THESE TO THEIR LOCAL SETUP !!!
IMAGENET_TRAIN = "/path/to/ibex/train"
IMAGENET_VAL = "/path/to/ibex/val"
IMAGENET_C = "/path/to/ibex/ImageNet-C" # Base directory for ImageNet-C

PROJECT        = "LearnableAugMixSeverityJSD_Type1_IMNET" # WandB Project Name
CKPT_DIR_LAUGMIX_JSD = "./checkpoints_learnable_augmix_jsd_type1" # Default checkpoint dir

# Normalization constants, reshaped for broadcasting (B,C,H,W)
MEAN = torch.tensor([0.485,0.456,0.406], dtype=torch.float).view(1,3,1,1)
STD  = torch.tensor([0.229,0.224,0.225], dtype=torch.float).view(1,3,1,1)

CORRUPTIONS = ["gaussian_noise","shot_noise","impulse_noise","defocus_blur","glass_blur","motion_blur","zoom_blur","snow","frost","fog","brightness","contrast","elastic_transform","pixelate","jpeg_compression"]

# ────────────────── AugMix Helper Operations (PIL-based) ───────────────────
# These functions operate on single PIL Images and take an op_level (0-10).
def int_parameter(level, maxval): return int(level * maxval / 10)
def float_parameter(level, maxval): return float(level) * maxval / 10.
def autocontrast(pil_img, _): return ImageOps.autocontrast(pil_img)
def equalize(pil_img, _): return ImageOps.equalize(pil_img)
def posterize(pil_img, level): return ImageOps.posterize(pil_img, 4 - int_parameter(level, 4))
def rotate(pil_img, level):
  deg = int_parameter(level, 30); return pil_img.rotate(deg if random.random() < 0.5 else -deg, resample=Image.BILINEAR)
def solarize(pil_img, level): return ImageOps.solarize(pil_img, 256 - int_parameter(level, 256))
def shear_x(pil_img, level):
  lvl = float_parameter(level, 0.3); return pil_img.transform(pil_img.size, Image.AFFINE, (1, lvl if random.random()<0.5 else -lvl, 0, 0, 1, 0), resample=Image.BILINEAR)
def shear_y(pil_img, level):
  lvl = float_parameter(level, 0.3); return pil_img.transform(pil_img.size, Image.AFFINE, (1, 0, 0, lvl if random.random()<0.5 else -lvl, 1, 0), resample=Image.BILINEAR)
def translate_x(pil_img, level):
  lvl = int_parameter(level, pil_img.size[0]/3); return pil_img.transform(pil_img.size, Image.AFFINE, (1,0,lvl if random.random()<0.5 else -lvl,0,1,0), resample=Image.BILINEAR)
def translate_y(pil_img, level):
  lvl = int_parameter(level, pil_img.size[1]/3); return pil_img.transform(pil_img.size, Image.AFFINE, (1,0,0,0,1,lvl if random.random()<0.5 else -lvl), resample=Image.BILINEAR)
def color(pil_img, level): return ImageEnhance.Color(pil_img).enhance(float_parameter(level, 1.8) + 0.1)
def contrast(pil_img, level): return ImageEnhance.Contrast(pil_img).enhance(float_parameter(level, 1.8) + 0.1)
def brightness(pil_img, level): return ImageEnhance.Brightness(pil_img).enhance(float_parameter(level, 1.8) + 0.1)
def sharpness(pil_img, level): return ImageEnhance.Sharpness(pil_img).enhance(float_parameter(level, 1.8) + 0.1)
# List of available PIL-based augmentation functions for AugMix chains
available_pil_augmentations = [autocontrast, equalize, posterize, rotate, solarize, shear_x, shear_y, translate_x, translate_y, color, contrast, brightness, sharpness]

# The tensor_ops_list was defined in a previous version but is not used here as Augmenter works on PIL.
# Keeping available_tensor_augmentations commented out for now. If you intend to use tensor ops,
# the Augmenter logic would need to change to accept tensors.
# available_tensor_augmentations = [ ... ]


# ───────────────────────── data loaders ──────────────────────
def loaders(batch_size_arg, num_workers_arg, aug_type, current_device_type):
    """
    Configures DataLoaders.
    If aug_type is 'learnable_augmix_jsd', the training DataLoader will yield
    (list of PIL Images, tensor of labels) using a custom collate_fn.
    Otherwise, it yields (batch_tensor, batch_labels_tensor).
    """
    if aug_type == "learnable_augmix_jsd":
        # For LearnableAugMix (nn.Module), dataset provides PIL images
        train_transforms = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip()])
        collate_fn_train = augmix_pil_collate # Custom collate for list of PIL images
    elif aug_type == "none":
        train_transforms = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor()])
        collate_fn_train = None # Default collate handles (Tensor, Tensor)
    else:
        raise ValueError(f"Unsupported aug_type for loader setup: {aug_type}")

    val_transforms_tensor = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    if not os.path.isdir(IMAGENET_TRAIN): raise FileNotFoundError(f"Train dir: {IMAGENET_TRAIN}")
    if not os.path.isdir(IMAGENET_VAL): raise FileNotFoundError(f"Val dir: {IMAGENET_VAL}")
    
    train_ds = dset.ImageFolder(IMAGENET_TRAIN, train_transforms)
    val_ds = dset.ImageFolder(IMAGENET_VAL, val_transforms_tensor)
    
    use_persistent = num_workers_arg > 0
    pin_memory_flag = (current_device_type == "gpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size_arg, shuffle=True,
                              num_workers=num_workers_arg, pin_memory=pin_memory_flag,
                              persistent_workers=use_persistent, collate_fn=collate_fn_train)
    val_loader = DataLoader(val_ds, batch_size=batch_size_arg, shuffle=False,
                            num_workers=num_workers_arg, pin_memory=pin_memory_flag,
                            persistent_workers=use_persistent)
    return train_loader, val_loader, val_transforms_tensor

def augmix_pil_collate(batch):
    """Collate function for DataLoader when dataset yields (PIL Image, label) tuples."""
    images = [item[0] for item in batch] # Creates a list of PIL Images
    labels = torch.tensor([item[1] for item in batch]) # Collates labels into a tensor
    return images, labels # Returns (list_of_PIL_images, tensor_of_labels)

def loaders_imagenet_c(transforms_arg, batch_size_arg, num_workers_arg, current_device_type):
    """Creates DataLoaders for ImageNet-C evaluation."""
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
            loader_c = DataLoader(ds_c, batch_size=batch_size_arg, shuffle=False,
                                  num_workers=num_workers_arg, pin_memory=pin_memory_flag,
                                  persistent_workers=use_persistent)
            dataloaders_c.append((corr, sev, loader_c))
    if not dataloaders_c: print(f"Warning: No ImageNet-C data loaded from {IMAGENET_C}.")
    return dataloaders_c

# ─────────────────── Augmenter Base Class ───────────────────
class Augmenter(nn.Module):
    """Base class for augmenters. Defines interface for forward and kl."""
    requires_labels = False # Default; augmenters like Mixup/CutMix override this.
    def __init__(self): super().__init__()
    def forward(self, x_input_batch, y_input_batch=None): raise NotImplementedError
    def kl(self): # Default KL divergence is 0.
        dev = 'cpu'; param_found = False
        # Try to infer device from known learnable parameter names or any parameter/buffer
        if hasattr(self, 'log_severity_mu'): dev = self.log_severity_mu.device; param_found=True
        elif hasattr(self, 'dummy_param'): dev = self.dummy_param.device; param_found=True # For NoAug
        # Add other specific learnable param names here if new augmenters are created
        if not param_found:
            try: dev = next(self.parameters()).device
            except StopIteration:
                try: dev = next(self.buffers()).device
                except StopIteration: pass
        return torch.tensor(0.0, device=dev)

# ─────────────────── No Augmentation Wrapper ──────────────────
class NoAug(Augmenter):
    """Augmenter that applies no transformation (identity).
       If input is PIL list, converts to tensor. If input is tensor, passes through.
    """
    def __init__(self):
        super().__init__()
        self._to_tensor = T.ToTensor() # For converting PIL images
        self.register_buffer("dummy_param", torch.empty(0)) # For device detection in kl()
    
    def forward(self, x_input_batch, y_input_batch=None): # y_input_batch is ignored
        if isinstance(x_input_batch, list) and all(isinstance(im, Image.Image) for im in x_input_batch):
            # Converts list of PIL images to a stacked tensor on CPU.
            # The training_step will then move this tensor to the correct GPU device.
            return torch.stack([self._to_tensor(im) for im in x_input_batch])
        elif isinstance(x_input_batch, torch.Tensor):
            return x_input_batch # Pass through if already a tensor
        else: raise TypeError(f"NoAug expects list of PIL Images or a Tensor, got {type(x_input_batch)}")

# ─── Learnable AugMix Severity + JSD Augmenter (as nn.Module) ───
class LearnableAugMixSeverityJSDAugmenter(Augmenter):
    """
    Implements AugMix with JSD. The 'aug_severity' for the base operations
    is controlled by a learnable distribution (LogNormal).
    The augmenter itself is an nn.Module.
    Input: list of PIL Images. Output: Tuple of 3 Tensors (orig, aug1, aug2) on module's device.
    """
    def __init__(self, mixture_width=3, mixture_depth=-1,
                 initial_aug_severity=3.0, prior_severity_std=1.0,
                 prior_severity_mean_offset=0.0):
        super().__init__()
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth # -1 for random depth 1-3
        
        # Clamp initial severity for log stability
        clamped_initial_severity = np.clip(float(initial_aug_severity), 0.1, 10.0)
        
        # Learnable parameters for the distribution of log(aug_severity)
        # These nn.Parameters will be moved to the correct device by PyTorch Lightning.
        self.log_severity_mu = nn.Parameter(torch.log(torch.tensor(clamped_initial_severity, dtype=torch.float)))
        self.log_severity_log_sigma = nn.Parameter(torch.log(torch.tensor(0.1, dtype=torch.float))) # Small initial variance

        # Prior distribution parameters for log(aug_severity)
        prior_log_sev_mu_val = torch.log(torch.tensor(clamped_initial_severity, dtype=torch.float)) + prior_severity_mean_offset
        self.register_buffer("prior_log_severity_mu_buffer", prior_log_sev_mu_val)
        self.register_buffer("prior_log_severity_std_buffer", torch.tensor(prior_severity_std, dtype=torch.float))
        
        self._pil_to_tensor = T.ToTensor() # Converts PIL to CPU Tensor in [0,1]
        self._last_sampled_severity = clamped_initial_severity # For logging

    def _apply_op_chain(self, image_pil, current_aug_severity):
        """Applies a chain of PIL augmentations to a single PIL image."""
        image_aug = image_pil.copy()
        # Determine depth of augmentation chain
        depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(available_pil_augmentations) # Choose a PIL-based op
            # Sample a level for this specific operation instance.
            # The op helper functions (int_parameter, float_parameter) expect level in [0, 10].
            # current_aug_severity is also on a similar scale (e.g., 0.1 to 10).
            op_level_for_helper = np.random.uniform(0, current_aug_severity)
            image_aug = op(image_aug, op_level_for_helper)
        return self._pil_to_tensor(image_aug) # Convert augmented PIL to CPU Tensor

    def forward(self, x_pil_batch, y_input_batch=None): # x_pil_batch is a list of PIL Images
        # Determine the target device from one of the module's parameters.
        # PyTorch Lightning ensures nn.Parameters (and buffers) are on the correct device.
        target_device = self.log_severity_mu.device
        
        # Sample a single augmentation severity for the entire batch for efficiency.
        # Could be extended to per-image sampling if needed.
        epsilon_sev = torch.randn(1, device=target_device) # Sample noise on target device
        log_sev_sample = self.log_severity_mu + torch.exp(self.log_severity_log_sigma) * epsilon_sev
        current_aug_severity_param = torch.exp(log_sev_sample).item() # Get as Python float
        # Clamp the sampled severity to a practical range for augmentation operations.
        current_aug_severity_clamped = np.clip(current_aug_severity_param, 0.1, 10.0)
        self._last_sampled_severity = current_aug_severity_clamped # Store for logging

        x_orig_tensors, x_aug1_tensors, x_aug2_tensors = [], [], []
        for pil_img_orig in x_pil_batch:
            # Convert original PIL to CPU Tensor, then move to target_device.
            orig_tensor = self._pil_to_tensor(pil_img_orig).to(target_device)
            x_orig_tensors.append(orig_tensor)
            
            # AugMix View 1
            # Mixing weights for augmentation chains (Dirichlet)
            ws1_np = np.float32(np.random.dirichlet([1.0]*self.mixture_width))
            ws1 = torch.from_numpy(ws1_np).to(target_device)
            # Mixing coefficient for (original vs. mixed_augmentations) (Beta)
            m1 = torch.tensor(np.float32(np.random.beta(1.0,1.0)), device=target_device)
            
            # Accumulate weighted augmented chains
            mix1_accumulator = torch.zeros_like(orig_tensor, device=target_device) # Create on target_device
            for i in range(self.mixture_width):
                # _apply_op_chain returns a CPU tensor, move it to target_device
                aug_chain_tensor = self._apply_op_chain(pil_img_orig, current_aug_severity_clamped).to(target_device)
                mix1_accumulator += ws1[i] * aug_chain_tensor
            # Final mix for view 1
            x_aug1_tensors.append((1-m1)*orig_tensor + m1*mix1_accumulator)
            
            # AugMix View 2 (independent random parameters)
            ws2_np = np.float32(np.random.dirichlet([1.0]*self.mixture_width))
            ws2 = torch.from_numpy(ws2_np).to(target_device)
            m2 = torch.tensor(np.float32(np.random.beta(1.0,1.0)), device=target_device)
            mix2_accumulator = torch.zeros_like(orig_tensor, device=target_device)
            for i in range(self.mixture_width):
                aug_chain_tensor = self._apply_op_chain(pil_img_orig, current_aug_severity_clamped).to(target_device)
                mix2_accumulator += ws2[i] * aug_chain_tensor
            x_aug2_tensors.append((1-m2)*orig_tensor + m2*mix2_accumulator)
            
        # Stack lists of tensors into batch tensors. All should be on target_device.
        return torch.stack(x_orig_tensors), torch.stack(x_aug1_tensors), torch.stack(x_aug2_tensors)

    def kl(self): # KL for the learned log_severity distribution
        q_mu = self.log_severity_mu
        q_log_sigma = self.log_severity_log_sigma
        q_sigma_sq = torch.exp(2 * q_log_sigma) # learned variance sigma_q^2
        
        p_mu = self.prior_log_severity_mu_buffer
        p_sigma = self.prior_log_severity_std_buffer
        p_sigma_sq = p_sigma**2 # prior variance

        # KL(q || p) = log(sigma_p/sigma_q) + (sigma_q^2 + (mu_q - mu_p)^2)/(2*sigma_p^2) - 0.5
        kl_div = (p_sigma.log() - q_log_sigma) + \
                 (q_sigma_sq + (q_mu - p_mu)**2) / (2 * p_sigma_sq) - 0.5
        return kl_div

# ───────────────────── Lightning module ──────────────────────
class LitImageNetModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-4, beta_network_reg=0.01,
                 augmentation_type="learnable_augmix_jsd",
                 augmix_mixture_width=3, augmix_mixture_depth=-1,
                 initial_aug_severity=3.0, beta_jsd=12.0,
                 beta_augmenter_reg=1.0, load_pretrained_for_no_aug_eval=False,
                 prior_severity_std_learnable_aug=1.0, 
                 prior_severity_mean_offset_learnable_aug=0.0):
        super().__init__()
        self.save_hyperparameters() # Saves all __init__ args
        net_weights = models.ResNet50_Weights.IMAGENET1K_V2
        self.net = models.resnet50(weights=net_weights)
        if not self.hparams.load_pretrained_for_no_aug_eval:
            self.net.fc = nn.Linear(self.net.fc.in_features, 1000)

        aug_type = self.hparams.augmentation_type
        if aug_type == "learnable_augmix_jsd":
            self.augmenter = LearnableAugMixSeverityJSDAugmenter(
                mixture_width=self.hparams.augmix_mixture_width,
                mixture_depth=self.hparams.augmix_mixture_depth,
                initial_aug_severity=self.hparams.initial_aug_severity,
                prior_severity_std=self.hparams.prior_severity_std_learnable_aug,
                prior_severity_mean_offset=self.hparams.prior_severity_mean_offset_learnable_aug
            )
        elif aug_type == "none": self.augmenter = NoAug()
        else: raise ValueError(f"Unsupported aug_type: {aug_type}")
        
        # mean_tf and std_tf are already shaped (1,C,1,1) globally
        self.register_buffer('mean_tf',MEAN) 
        self.register_buffer('std_tf',STD)

    def forward(self, x): # x is on self.device
        # Buffers are moved to model's device by PL.
        normalized_x = (x - self.mean_tf) / self.std_tf
        return self.net(normalized_x)

    def _calculate_jsd_loss(self, logits_list):
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        # Clamp probabilities for numerical stability in KL divergence
        probs_clamped_list = [torch.clamp(p, 1e-7, 1.0 - 1e-7) for p in probs_list] # Adjusted clamp
        
        p_mixture = torch.stack(probs_clamped_list, dim=0).mean(dim=0)
        # Clamping p_mixture might not be strictly necessary if individual ones are clamped
        # and averaging preserves sum-to-1, but can be a safeguard.
        # p_mixture_clamped = torch.clamp(p_mixture, 1e-7, 1.0 - 1e-7)

        jsd = 0.0
        for p_single_view_clamped in probs_clamped_list:
            # Calculate KL(P_single || P_mixture)
            # F.kl_div expects (input_log_probs, target_probs) when log_target=False
            kl_div_val = F.kl_div(input=p_single_view_clamped.log(), # P.log()
                                  target=p_mixture, # Q (use unclamped p_mixture if p_single are well-behaved)
                                  reduction='batchmean', 
                                  log_target=False) # target is not log_probs
            jsd += kl_div_val
        return jsd / len(probs_clamped_list)

    def training_step(self, batch, batch_idx):
        pil_list, labels_orig_cpu = batch # From custom collate_fn if learnable_augmix_jsd
        B = len(pil_list) if isinstance(pil_list, list) else pil_list.shape[0]
        labels_orig = labels_orig_cpu.to(self.device) # Move labels to model's device
        
        jsd_loss = torch.tensor(0.0, device=self.device)
        aug_kl_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.augmentation_type == "learnable_augmix_jsd":
            # self.augmenter is an nn.Module already on self.device.
            # Its forward method now ensures output tensors are on its device.
            x_orig, x_aug1, x_aug2 = self.augmenter(pil_list) # Input is list of PIL
            # x_orig, x_aug1, x_aug2 should now be on self.device
            
            l_o,l_a1,l_a2 = self(x_orig), self(x_aug1), self(x_aug2)
            ce_loss = F.cross_entropy(l_o, labels_orig)
            jsd_loss = self._calculate_jsd_loss([l_o, l_a1, l_a2])
            aug_kl_loss = self.augmenter.kl() # KL for severity params, on self.device
            # For KL, the original paper does not divide by batch size, but often it's done for other KL terms.
            # Let's try without /B for aug_kl_loss first, as it's a scalar parameter KL.
            # Or, if we want beta_augmenter_reg to be more like a per-sample weight, divide by B.
            # Sticking to /B for now as it makes beta_augmenter_reg less dependent on batch_size.
            total_loss = ce_loss + self.hparams.beta_jsd*jsd_loss + self.hparams.beta_augmenter_reg*aug_kl_loss/B 
            acc = (l_o.argmax(dim=1) == labels_orig).float().mean()
        elif self.hparams.augmentation_type == "none":
            # If input is list of PIL, NoAug converts to CPU tensor.
            # If input is already tensor (e.g. from default Dataloader if 'none' logic changes), NoAug passes it.
            x_tensor_processed = self.augmenter(pil_list).to(self.device)
            logits = self(x_tensor_processed); ce_loss = F.cross_entropy(logits, labels_orig)
            total_loss = ce_loss; acc = (logits.argmax(dim=1) == labels_orig).float().mean()
        else: raise NotImplementedError(f"Training for {self.hparams.augmentation_type} not set up.")

        # Safely get learning rate
        lr_val = 0.0
        if self.trainer and self.trainer.optimizers and self.trainer.optimizers[0].param_groups:
             lr_val = self.trainer.optimizers[0].param_groups[0]['lr']

        log_data = {"trL":total_loss,"trAcc":acc,"trCE":ce_loss,"trJSD":jsd_loss,"trAugKL":aug_kl_loss,"lr":lr_val}
        if isinstance(self.augmenter, LearnableAugMixSeverityJSDAugmenter):
            log_data["log_sev_mu"] = self.augmenter.log_severity_mu.item()
            log_data["log_sev_lsigma"] = self.augmenter.log_severity_log_sigma.item()
            log_data["samp_sev"] = getattr(self.augmenter, '_last_sampled_severity', -1.0)
        self.log_dict(log_data, prog_bar=True, on_step=True, on_epoch=False, batch_size=B)
        return total_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad(): # Apply torch.no_grad()
            x_tensor, y_raw = batch # x_tensor is on device by PL
            y = y_raw.to(self.device) # Ensure labels are on device
            logits=self(x_tensor)
            loss=F.cross_entropy(logits,y)
            acc=(logits.argmax(dim=1)==y).float().mean()
        self.log_dict({"val_loss":loss,"val_acc":acc},prog_bar=True,on_epoch=True,sync_dist=True, batch_size=x_tensor.shape[0])
        return {"val_loss":loss,"val_acc":acc}

    def configure_optimizers(self):
        net_p=[p for n,p in self.named_parameters() if n.startswith("net.")and p.requires_grad]
        aug_p=[p for n,p in self.named_parameters() if n.startswith("augmenter.")and p.requires_grad]
        groups=[]
        if not net_p: raise ValueError("No net params requiring grad.")
        groups.append({"params":net_p,"lr":self.hparams.learning_rate,"weight_decay":self.hparams.beta_network_reg})
        if aug_p:
            aug_lr = self.hparams.learning_rate * 10 # Augmenter params might need different LR
            groups.append({"params":aug_p,"lr":aug_lr,"weight_decay":0.0})
            # Log only on rank 0 to avoid cluttered logs in DDP
            if not hasattr(self, 'trainer') or self.trainer.global_rank == 0:
                print(f"Augmenter parameters included in optimizer with LR: {aug_lr}")
        elif self.hparams.augmentation_type=="learnable_augmix_jsd":
             # This warning is more critical if aug_p is empty when it's expected
             if not hasattr(self, 'trainer') or self.trainer.global_rank == 0:
                print("Warning: LearnableAugMix specified, but NO augmenter parameters found for optimizer. Check requires_grad.")
        
        opt=torch.optim.AdamW(groups) # AdamW takes list of param groups
        sch=torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10,T_mult=2,eta_min=self.hparams.learning_rate/100.0)
        return {"optimizer":opt,"lr_scheduler":{"scheduler":sch,"interval":"epoch"}}

# ───────────────────── main routine ──────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ImageNet Training with Learnable AugMix Severity + JSD")
    parser.add_argument('--epochs',type=int,default=90);parser.add_argument('--batch',type=int,default=128) # Batch size might need adjustment due to memory
    parser.add_argument('--workers',type=int,default=8);parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--gpus',type=int,default=-1);parser.add_argument('--strategy',type=str,default=None)
    parser.add_argument('--precision',type=str,default="16",help="PyTorch Lightning precision: '16','bf16','32','64'.")
    parser.add_argument('--ckpt_dir_base',default=None);parser.add_argument('--eval_only',action='store_true')
    parser.add_argument('--ckpt_path',default=None);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--aug_type',type=str,default="learnable_augmix_jsd",choices=["learnable_augmix_jsd","none"])
    parser.add_argument('--augmix_mixture_width',type=int,default=3)
    parser.add_argument('--augmix_mixture_depth',type=int,default=-1, help="Max depth of augmentation chain, -1 for random 1-3.")
    parser.add_argument('--initial_aug_severity',type=float,default=3.0, help="Initial mean for severity distribution.")
    # Added arguments for prior distribution of learnable severity
    parser.add_argument('--prior_severity_std_learnable_aug',type=float,default=1.0, help="Stddev for prior on log_severity for learnable augmenter.")
    parser.add_argument('--prior_severity_mean_offset_learnable_aug', type=float, default=0.0, help="Offset for prior mean on log_severity.")
    parser.add_argument('--beta_jsd',type=float,default=12.0, help="Weight for JSD loss.")
    parser.add_argument('--beta_augmenter_reg',type=float,default=1.0, help="Weight for KL divergence of augmenter params.")
    parser.add_argument('--beta_net',type=float,default=0.01, help="Weight decay for network parameters.")
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--no_wandb',action='store_true')
    parser.add_argument('--matmul_precision',type=str,default='medium', help="Torch matmul precision: 'medium' or 'high'.")
    parser.add_argument('--grad_clip_val', type=float, default=1.0, help="Gradient clipping value (0 for no clipping).")
    args = parser.parse_args()

    if torch.cuda.is_available():torch.set_float32_matmul_precision(args.matmul_precision)
    pl.seed_everything(args.seed,workers=True)

    dev_t,num_dev,strat_val="cpu",1,"auto" # Default to auto for single device strategy
    if args.gpus==0:pass
    elif torch.cuda.is_available():
        avail=torch.cuda.device_count();actual=avail if args.gpus==-1 else min(args.gpus,avail)
        if actual>0:
            dev_t,num_dev = "gpu", actual
            if args.strategy: strat_val = args.strategy
            elif actual > 1: # DDP for multi-GPU
                # More robust way to try DDPStrategy with find_unused_parameters
                try:
                    from pytorch_lightning.strategies import DDPStrategy
                    # For learnable augmenter params, True might be needed if not all ranks use them identically every step,
                    # but can be slower. False if confident all are used.
                    strat_val = DDPStrategy(find_unused_parameters=False) # Changed to False for potential speedup
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                         print("Using PL DDPStrategy(find_unused_parameters=False).")
                except (ImportError, TypeError, AttributeError):
                    strat_val = "ddp" # Fallback
                    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                        print("Using older PL DDP strategy string 'ddp'.")
            # else strat_val remains "auto" for single GPU
        else: print("Warn: GPUs req but none avail. Using CPU.")
    else: print("Warn: CUDA not avail. Using CPU.")
    
    effective_strategy_name = strat_val.strategy_name if hasattr(strat_val, 'strategy_name') else strat_val
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"Using {num_dev} {dev_t}(s). Effective Strategy: {effective_strategy_name}")

    train_loader_final,val_loader_cfg,val_tf=loaders(args.batch,args.workers, args.aug_type, dev_t)
    imnet_c_loaders=loaders_imagenet_c(val_tf,args.batch,args.workers, dev_t)
    
    model_hparams={ "learning_rate":args.lr, "beta_network_reg":args.beta_net,
                     "augmentation_type":args.aug_type, "augmix_mixture_width":args.augmix_mixture_width,
                     "augmix_mixture_depth":args.augmix_mixture_depth, "initial_aug_severity":args.initial_aug_severity,
                     "beta_jsd":args.beta_jsd, "beta_augmenter_reg":args.beta_augmenter_reg,
                     "prior_severity_std_learnable_aug": args.prior_severity_std_learnable_aug,
                     "prior_severity_mean_offset_learnable_aug": args.prior_severity_mean_offset_learnable_aug}

    ckpt_b=args.ckpt_dir_base if args.ckpt_dir_base else CKPT_DIR_LAUGMIX_JSD
    if args.aug_type=="none"and not args.ckpt_dir_base:ckpt_b="./checkpoints_no_aug_eval"
    final_ckpt_dir=os.path.join(ckpt_b,args.aug_type);os.makedirs(final_ckpt_dir,exist_ok=True)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0: print(f"Ckpts/logs dir: {final_ckpt_dir}")

    wandb_log=None
    if not args.no_wandb:
        # Ensure WandB is initialized only on rank 0 in DDP
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            parts=[PROJECT,args.aug_type]
            if args.aug_type=="learnable_augmix_jsd":parts.extend([f"w{args.augmix_mixture_width}",f"d{args.augmix_mixture_depth}",f"is{args.initial_aug_severity}",f"priorS{args.prior_severity_std_learnable_aug}",f"priorMO{args.prior_severity_mean_offset_learnable_aug}",f"jsd{args.beta_jsd}",f"kl{args.beta_augmenter_reg}"])
            parts.append(f"s{args.seed}");
            if args.eval_only:parts.append("eval")
            name="_".join(map(str,parts));wandb_log=WandbLogger(project=PROJECT,name=name,save_dir=final_ckpt_dir)
            wandb_log.log_hyperparams(vars(args)) # Log CLI args
        else: # For non-rank 0 processes in DDP, create a dummy logger or disable
            wandb_log = WandbLogger(project=PROJECT, name="disabled_rank", offline=True, mode="disabled")


    model,trainer_fit=None,None # trainer_fit for the training Trainer instance
    if args.eval_only:
        load_pretrained = True if args.aug_type=="none" else False
        # Ensure all hparams for LitImageNetModel are available for instantiation
        eval_hparams = {**model_hparams, "load_pretrained_for_no_aug_eval": load_pretrained}
        if args.ckpt_path:
            if not os.path.exists(args.ckpt_path):raise FileNotFoundError(f"Eval ckpt:{args.ckpt_path}")
            model=LitImageNetModel.load_from_checkpoint(args.ckpt_path,map_location='cpu',**eval_hparams)
        elif args.aug_type=="none": model=LitImageNetModel(**eval_hparams)
        else:raise ValueError("Eval needs --ckpt_path or (aug_type 'none').")
    else: # Training mode
        if args.aug_type=="none":
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0: print("Err:No training for 'none'.")
            return
        
        # Pass all necessary hparams for model creation if not resuming
        # If resuming, these hparams can override checkpoint hparams
        train_model_hparams = {**model_hparams, "load_pretrained_for_no_aug_eval": False}

        ckpt_cb=ModelCheckpoint(dirpath=final_ckpt_dir,monitor="val_acc",mode="max",filename=f"{args.aug_type}-best-{{epoch:02d}}-{{val_acc:.3f}}",save_top_k=1,save_last=True) # save_last=True is good for recovery
        lr_cb=LearningRateMonitor(logging_interval='epoch')
        
        trainer_fit_params = {"max_epochs":args.epochs,"accelerator":dev_t,"devices":num_dev,"strategy":strat_val,
                              "precision":args.precision,"logger":wandb_log if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else False, # Log only on rank 0
                              "callbacks":[ckpt_cb,lr_cb] if (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0) else [lr_cb], # Checkpoints only on rank 0
                              "log_every_n_steps":50,"gradient_clip_val":args.grad_clip_val if args.grad_clip_val > 0 else None}
        trainer_fit=pl.Trainer(**trainer_fit_params)
        
        ckpt_res=args.ckpt_path if args.resume and args.ckpt_path else None
        if ckpt_res and not os.path.exists(ckpt_res):raise FileNotFoundError(f"Resume ckpt:{ckpt_res}")
        
        if ckpt_res:
            if trainer_fit.global_rank==0: print(f"Resuming from {ckpt_res}.")
            m_train = LitImageNetModel.load_from_checkpoint(ckpt_res, **train_model_hparams)
        else:
            if trainer_fit.global_rank==0: print("New training.")
            m_train = LitImageNetModel(**train_model_hparams)
        
        # Watch model only on rank 0
        if wandb_log and wandb_log.experiment.mode != "disabled" and trainer_fit.global_rank==0:
            wandb_log.watch(m_train,log='all',log_freq=100)
            
        trainer_fit.fit(m_train,train_loader_final,val_loader_cfg,ckpt_path=ckpt_res)
        
        # Load best model on rank 0 after training for evaluation
        if trainer_fit.global_rank==0 or num_dev==1 :
            best_p=ckpt_cb.best_model_path
            if best_p and os.path.exists(best_p):
                print(f"Loading best model from {best_p} for final evaluation.")
                model=LitImageNetModel.load_from_checkpoint(best_p,map_location='cpu',**train_model_hparams)
            else: # Fallback to last model state if best_model_path is not found
                print("Warning: Best model checkpoint not found or specified. Using model state at end of training.")
                model=m_train
    
    # Synchronize before evaluation if DDP was used
    if num_dev > 1 and torch.distributed.is_initialized():
        torch.distributed.barrier()
        # If only rank 0 loaded the model for eval, other ranks might need to skip eval or get the model
        # For simplicity, let's assume if model is None on non-rank0, they exit.
        # A more robust way would be to broadcast the best_model_path from rank 0.
        if model is None and trainer_fit and trainer_fit.global_rank != 0:
             print(f"Rank {torch.distributed.get_rank()} has no model for evaluation, exiting.")
             if wandb_log and wandb_log.experiment.mode != "disabled": wandb.finish(exit_code=1)
             return


    if model is None: # Check if model is loaded (especially for non-rank 0 in DDP if not handled above)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
             print("No model available for evaluation on this rank. Exiting.")
        if wandb_log and wandb_log.experiment.mode != "disabled": wandb.finish(exit_code=1 if model is None else 0)
        return
    
    model.cpu() # Ensure model is on CPU before creating single-device eval_trainer
    
    # Final evaluation on a single device (preferably GPU if available, else CPU)
    final_eval_accel,final_eval_devs,final_eval_strat_str = dev_t if dev_t == "gpu" else "cpu", 1, "auto"
    eval_prec="32"if args.precision not in["32","64","32-true","64-true"]else args.precision
    
    eval_trainer_params = {"accelerator":final_eval_accel,"devices":final_eval_devs,
                           "precision":eval_prec,"logger":False,"strategy":final_eval_strat_str}
    eval_trainer=pl.Trainer(**eval_trainer_params)

    # Perform evaluation only on rank 0 (or if not distributed)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(f"\n--- Validating Clean ({model.hparams.augmentation_type}) p={eval_prec} ---")
        val_r=eval_trainer.validate(model,val_loader_cfg,verbose=False)
        clean_acc,clean_loss=(val_r[0]['val_acc'],val_r[0]['val_loss'])if val_r and len(val_r)>0 else(0.0,float('nan'))
        print(f"Clean Acc@1:{clean_acc*100:5.2f}%,Loss:{clean_loss:.4f}")
        if wandb_log and wandb_log.experiment.mode != "disabled":
            wandb_log.log_metrics({"final_clean_acc":clean_acc,"final_clean_loss":clean_loss})

        print(f"\n--- Eval ImageNet-C ({model.hparams.augmentation_type}) p={eval_prec} ---")
        if not imnet_c_loaders:print("Skip ImageNet-C:no loaders.")
        else:
            c_err_s,c_mean_e,all_c_a={},{},[]
            for n,s,l_c in imnet_c_loaders:
                print(f"Eval:{n} s{s}...");r_c=eval_trainer.validate(model,l_c,verbose=False)
                a_c=(r_c[0]['val_acc'] if r_c and len(r_c)>0 else float('nan'))
                e_c=1.0-a_c if not math.isnan(a_c)else float('nan')
                if not math.isnan(a_c):all_c_a.append(a_c)
                if n not in c_err_s:c_err_s[n]=[float('nan')]*5
                if 1<=s<=5:c_err_s[n][s-1]=e_c;print(f"{n:20s} s{s}: Acc {a_c*100:5.2f}% | Err {e_c*100:5.2f}%")
            l_corr_avg_e=[]
            for c_name,e_l in sorted(c_err_s.items()):
                v_e=[e_val for e_val in e_l if not math.isnan(e_val)]
                if len(v_e)==5:avg_e=np.mean(v_e);c_mean_e[c_name]=avg_e;l_corr_avg_e.append(avg_e);print(f"{c_name:20s}:AvgErr {avg_e*100:5.2f}%")
                else:c_mean_e[c_name]=float('nan');print(f"{c_name:20s}:AvgErr NaN (got {len(v_e)})")
            mCE=np.mean(l_corr_avg_e)if l_corr_avg_e else float('nan')
            m_c_a=np.mean(all_c_a)if all_c_a else float('nan')
            print(f"\nImNet-C mCE:{mCE*100:5.2f}%");print(f"ImNet-C MeanAcc:{m_c_a*100:5.2f}%")
            if wandb_log and wandb_log.experiment.mode != "disabled":wandb_log.log_metrics({"final_imnet_c_mCE":mCE,"final_imnet_c_mean_acc":m_c_a})
            
            # Consolidate hparams for JSON from args and loaded model hparams
            js_h_cli = vars(args)
            js_h_model = dict(model.hparams)
            # Prioritize model's hparams if they exist, else use CLI args
            # This ensures that if a model is loaded, its actual trained hparams are logged
            final_logged_hparams = {**js_h_cli, **js_h_model}


            js_o={"cli_args_original_run":vars(args), # Keep original CLI args
                  "model_hparams_effective":final_logged_hparams, # Effective Hparams used/loaded
                  "clean_acc":clean_acc,"clean_loss":clean_loss,
                  "imnet_c_errors_sev":c_err_s,"imnet_c_mean_error_per_corruption":c_mean_e,
                  "mCE":mCE,"mean_accuracy_on_corrupted":m_c_a}
            
            js_f_parts=['results',str(final_logged_hparams.get('augmentation_type','unknown_aug'))]
            if final_logged_hparams.get('augmentation_type')=='learnable_augmix_jsd':
                js_f_parts.extend([f"w{final_logged_hparams.get('augmix_mixture_width',-1)}",
                                   f"d{final_logged_hparams.get('augmix_mixture_depth',-1)}",
                                   f"is{final_logged_hparams.get('initial_aug_severity',-1)}",
                                   f"priorS{final_logged_hparams.get('prior_severity_std_learnable_aug',-1)}",
                                   f"priorMO{final_logged_hparams.get('prior_severity_mean_offset_learnable_aug',-1)}",
                                   f"jsd{final_logged_hparams.get('beta_jsd',-1)}",
                                   f"kl{final_logged_hparams.get('beta_augmenter_reg',-1)}"])
            js_f_parts.append(f"s{args.seed}") # Use current run's seed for filename
            js_fname="_".join(map(str,js_f_parts))+".json";js_fp=os.path.join(final_ckpt_dir,js_fname)
            try:
                with open(js_fp,'w')as f:
                    def np_encoder(o):
                        if isinstance(o,(np.generic,np.ndarray))and o.size==1:return o.item()
                        if isinstance(o,(np.ndarray,torch.Tensor)):return o.tolist()
                        if isinstance(o,torch.device):return str(o)
                        # Fallback for other non-serializable types
                        return str(o)
                    json.dump(js_o,f,indent=2,default=np_encoder)
                print(f"Results saved to {js_fp}")
            except Exception as e:print(f"JSON save error: {e}. Object parts that might cause issues: {type(js_h)}")

    # Final WandB cleanup, only by rank 0 or single process
    if wandb_log and wandb_log.experiment and wandb_log.experiment.mode != "disabled":
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print(f"Finishing wandb run (rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0})...")
            wandb.finish(exit_code=0 if model is not None else 1)
    
    # Final barrier for DDP processes to sync before exiting
    if torch.distributed.is_initialized() and num_dev > 1:
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0: print("All DDP processes finished.")
    elif not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0 :
         print("\nRun complete on single process.")


if __name__=="__main__":
    main()