#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import torchmetrics

# Import custom modules
from archs.unet import UNetUnconditioned
from archs.wiener_deconv import WienerDeconvolution
from utils.loaders import StarRemovalDataset
from utils.transforms import make_transform_crop_rescale, make_blur_transform, generate_star_map_intensity
from utils.utils import plot_batch_v2

# Configure logger
log = logging.getLogger(__name__)


class HessianRegularization(nn.Module):
    """Simple placeholder for Hessian regularization."""
    def __init__(self):
        super(HessianRegularization, self).__init__()
        
    def forward(self, x):
        # Return a small constant value as a placeholder
        return torch.tensor(0.0, device=x.device)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Initialize optimizer based on config."""
    if cfg.training.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    elif cfg.training.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay, momentum=0.9)
    elif cfg.training.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer}")


def get_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Initialize learning rate scheduler based on config."""
    if cfg.training.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    elif cfg.training.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif cfg.training.scheduler == "none":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")


class MixedLoss(torch.nn.Module):
    def __init__(self, config):
        super(MixedLoss, self).__init__()

        self.config = config

        self.ssim = torchmetrics.image.StructuralSimilarityIndexMeasure(data_range=0.5, sigma=config.ssim_sigma).to("cuda" if torch.cuda.is_available() else "mps")
        self.reg = HessianRegularization()  # torchmetrics.image.TotalVariation().to("cuda" if torch.cuda.is_available() else "mps")

    def forward(self, x, y):
        ssim_loss = 1.0 - self.ssim(x, y)
        l1_loss = torch.nn.functional.l1_loss(x, y)
        reg_loss = self.reg(y)

        fft_x = torch.fft.fft2(x)
        fft_y = torch.fft.fft2(y)
        magnitude_x = torch.abs(fft_x)
        magnitude_y = torch.abs(fft_y)
        phase_x = torch.angle(fft_x.cpu()).to("mps")
        phase_y = torch.angle(fft_y.cpu()).to("mps")
        magnitude_loss = torch.nn.functional.l1_loss(magnitude_x, magnitude_y)
        phase_loss = 1.0 - torch.mean(torch.cos(phase_x - phase_y))

        total_loss = 0.0
        if self.config.ssim_weight is not None:
            total_loss += self.config.ssim_weight * ssim_loss
        if self.config.l1_weight is not None:
            total_loss += self.config.l1_weight * l1_loss
        if self.config.magnitude_weight is not None:
            total_loss += self.config.magnitude_weight * magnitude_loss
        if self.config.phase_weight is not None:
            total_loss += self.config.phase_weight * phase_loss
        if self.config.reg_weight is not None:
            total_loss += self.config.reg_weight * reg_loss

        losses = {
            "ssim_loss": self.config.ssim_weight * ssim_loss.item() if self.config.ssim_weight is not None else 0.0,
            "l1_loss": self.config.l1_weight * l1_loss.item() if self.config.l1_weight is not None else 0.0,
            "magnitude_loss": self.config.magnitude_weight * magnitude_loss.item() if self.config.magnitude_weight is not None else 0.0,
            "phase_loss": self.config.phase_weight * phase_loss.item() if self.config.phase_weight is not None else 0.0,
            "reg": self.config.reg_weight * reg_loss.item() if self.config.reg_weight is not None else 0.0,
        }

        return total_loss, losses


def get_loss_function(cfg: DictConfig) -> nn.Module:
    """Initialize loss function based on config."""
    if cfg.training.loss_fn == "mse":
        return nn.MSELoss()
    elif cfg.training.loss_fn == "l1":
        return nn.L1Loss()
    elif cfg.training.loss_fn == "huber":
        return nn.HuberLoss(delta=cfg.training.hyperparams.huber_delta)
    elif cfg.training.loss_fn == "ssim":
        try:
            from pytorch_msssim import SSIM
            return SSIM(data_range=1.0, size_average=True, channel=1, 
                        win_size=cfg.training.hyperparams.ssim_window_size)
        except ImportError:
            log.error("pytorch_msssim package not installed. Using MSELoss instead.")
            return nn.MSELoss()
    elif cfg.training.loss_fn == "fourier":
        # Get Fourier loss hyperparameters from config if available
        magnitude_weight = getattr(cfg.training.hyperparams, "fourier_magnitude_weight", 1.0)
        phase_weight = getattr(cfg.training.hyperparams, "fourier_phase_weight", 0.5)
        image_weight = getattr(cfg.training.hyperparams, "fourier_image_weight", 1.0)
        base_loss = getattr(cfg.training.hyperparams, "fourier_base_loss", "mse")
        huber_delta = getattr(cfg.training.hyperparams, "huber_delta", 0.1)
        
        return FourierLoss(
            base_loss=base_loss,
            magnitude_weight=magnitude_weight,
            phase_weight=phase_weight,
            image_weight=image_weight,
            huber_delta=huber_delta
        )
    elif cfg.training.loss_fn == "mixed":
        # Create a config object for MixedLoss from hyperparams
        mixed_loss_config = type('MixedLossConfig', (), {})()
        
        # Set the weights, defaulting to None if not specified
        mixed_loss_config.ssim_weight = getattr(cfg.training.hyperparams, "ssim_weight", 1.0)
        mixed_loss_config.l1_weight = getattr(cfg.training.hyperparams, "l1_weight", 1.0)
        mixed_loss_config.magnitude_weight = getattr(cfg.training.hyperparams, "magnitude_weight", 0.5)
        mixed_loss_config.phase_weight = getattr(cfg.training.hyperparams, "phase_weight", 0.5)
        mixed_loss_config.reg_weight = getattr(cfg.training.hyperparams, "reg_weight", 0.1)
        mixed_loss_config.ssim_sigma = getattr(cfg.training.hyperparams, "ssim_sigma", 1.5)
        
        return MixedLoss(mixed_loss_config)
    else:
        log.warning(f"Unsupported loss function: {cfg.training.loss_fn}. Using MixedLoss instead.")
        # Create a default config for MixedLoss
        default_config = type('MixedLossConfig', (), {})()
        default_config.ssim_weight = 1.0
        default_config.l1_weight = 1.0
        default_config.magnitude_weight = 0.5
        default_config.phase_weight = 0.5
        default_config.reg_weight = 0.1
        default_config.ssim_sigma = 1.5
        
        return MixedLoss(default_config)


def compute_metrics(pred: torch.Tensor, target: torch.Tensor, metrics: list) -> Dict[str, float]:
    """Compute evaluation metrics."""
    results = {}
    
    for metric in metrics:
        if metric == "mse":
            results["mse"] = nn.MSELoss()(pred, target).item()
        elif metric == "psnr":
            mse = torch.mean((pred - target) ** 2).item()
            results["psnr"] = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
        elif metric == "ssim":
            try:
                from pytorch_msssim import SSIM
                ssim_module = SSIM(data_range=1.0, size_average=True, channel=1)
                results["ssim"] = ssim_module(pred, target).item()
            except ImportError:
                results["ssim"] = 0.0
                log.warning("pytorch_msssim package not installed. SSIM calculation skipped.")
    
    return results


def save_checkpoint(
    model: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    epoch: int, 
    loss: float, 
    metrics: Dict[str, float], 
    checkpoint_path: str
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }, checkpoint_path)
    
    log.info(f"Checkpoint saved to {checkpoint_path}")


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    loss_fn: nn.Module, 
    device: torch.device, 
    mixed_precision: bool,
    scaler: torch.cuda.amp.GradScaler
) -> Dict[str, float]:
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    epoch_component_losses = {}
    
    # Training loop with progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for blurred, clean in pbar:
        # Move data to device
        blurred = blurred.to(device)
        clean = clean.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(blurred)
                output = blurred - output
                
                # Handle both regular loss functions and MixedLoss which returns a tuple
                if isinstance(loss_fn, MixedLoss):
                    loss, component_losses = loss_fn(output, clean)
                else:
                    loss = loss_fn(output, clean)
                    component_losses = {}
                
            # Check for NaN loss and skip the batch if detected
            if torch.isnan(loss):
                log.warning("NaN loss detected, skipping batch")
                continue
                
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Add gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward and backward pass
            output = model(blurred)
            output = blurred - output
            
            # Handle both regular loss functions and MixedLoss which returns a tuple
            if isinstance(loss_fn, MixedLoss):
                loss, component_losses = loss_fn(output, clean)
            else:
                loss = loss_fn(output, clean)
                component_losses = {}
            
            # Check for NaN loss and skip the batch if detected
            if torch.isnan(loss):
                log.warning("NaN loss detected, skipping batch")
                continue
                
            loss.backward()
            
            # Use higher max_norm for gradient clipping to allow faster learning
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
        
        # Update progress bar
        epoch_loss += loss.item()
        
        # Accumulate component losses if using MixedLoss
        for k, v in component_losses.items():
            if k not in epoch_component_losses:
                epoch_component_losses[k] = 0.0
            epoch_component_losses[k] += v
        
        # Show total loss and optionally component losses in progress bar
        if component_losses:
            pbar_dict = {"loss": f"{loss.item():.4f}"}
            for k, v in component_losses.items():
                pbar_dict[k] = f"{v:.4f}"
            pbar.set_postfix(pbar_dict)
        else:
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Calculate average losses
    num_batches = len(dataloader)
    avg_epoch_loss = epoch_loss / num_batches
    avg_component_losses = {k: v / num_batches for k, v in epoch_component_losses.items()}
    
    # Return average loss and component losses
    return {"total": avg_epoch_loss, **avg_component_losses}


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Print config
    log.info(f"Configuration: \n{OmegaConf.to_yaml(cfg)}")
    
    # Set random seed
    set_seed(cfg.training.seed)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    log.info(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    results_dir = os.path.join(cfg.results_dir, cfg.experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup data augmentation and transforms
    image_transform = make_transform_crop_rescale()
    blur_transform = make_blur_transform(sigma_range=(2.5, 4.0), sx_sy_max_ratio=(0.4, 1.6))
    
    # Create dataset and dataloader
    train_dataset = StarRemovalDataset(
        image_folder=cfg.dataset.train_data_path,
        psf_folder=cfg.dataset.psf_catalog.path,
        image_transform=image_transform,
        blur_transform=blur_transform,
        star_generator=generate_star_map_intensity,
        seed=cfg.training.seed
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    
    # Initialize models
    model = UNetUnconditioned(
        layer_sizes=cfg.model.layer_sizes,
        image_channels=cfg.model.image_channels,
    ).to(device)
    
    # Initialize optimizer, scheduler and loss function
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)
    loss_fn = get_loss_function(cfg)
    
    # Setup mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler() if cfg.training.mixed_precision else None
    
    # Initialize training variables
    start_epoch = 0
    best_val_loss = float('inf')  # Using this for best train loss now
    
    # Resume from checkpoint if specified
    if cfg.training.resume_from_checkpoint:
        checkpoint_path = cfg.training.resume_from_checkpoint
        if os.path.exists(checkpoint_path):
            log.info(f"Resuming from checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['loss']  # This will be the best train loss now
            log.info(f"Resuming from epoch {start_epoch}")
        else:
            log.warning(f"Checkpoint file not found: {checkpoint_path}")
    
    # Training loop
    log.info(f"Starting training for {cfg.training.num_epochs} epochs")
    
    for epoch in range(start_epoch, cfg.training.num_epochs):
        log.info(f"Epoch {epoch+1}/{cfg.training.num_epochs}")
        
        # Train one epoch
        train_loss_dict = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            mixed_precision=cfg.training.mixed_precision,
            scaler=scaler
        )
        
        # Extract total loss for checkpointing
        train_loss = train_loss_dict["total"]
        
        # Get samples from training set for visualization
        train_samples = []
        model.eval()
        with torch.no_grad():
            for batch_idx, (blurred, clean) in enumerate(train_loader):
                if batch_idx == 0:
                    # Move data to device
                    blurred = blurred.to(device)
                    clean = clean.to(device) 
                    
                    # Forward pass
                    output = model(blurred)
                    output = blurred - output
                    
                    train_samples = (blurred[:8].detach().cpu(), 
                                    output[:8].detach().cpu(), 
                                    clean[:8].detach().cpu())
                    break
        
        # Update learning rate
        scheduler.step()
        
        # Log metrics
        log_message = f"Train Loss: {train_loss:.4f}"
        # Add component losses to log if available
        for k, v in train_loss_dict.items():
            if k != "total":
                log_message += f", {k}: {v:.4f}"
        log.info(log_message)
        
        # Visualize results
        if epoch % 1 == 0:
            plot_batch_v2(
                train_samples[0],  # blurred
                train_samples[1],  # output
                train_samples[2],  # clean
                epoch=epoch,
                num_of_images=min(8, len(train_samples[0])),
                folder=results_dir,
            )
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.training.save_every == 0:
            checkpoint_path = os.path.join(
                cfg.training.checkpoint_dir, 
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                metrics={},  # No metrics for now
                checkpoint_path=checkpoint_path
            )
        
        # Save best model
        if train_loss < best_val_loss:  # Using train_loss instead of val_loss
            best_val_loss = train_loss
            best_checkpoint_path = os.path.join(
                cfg.training.checkpoint_dir, 
                "best_model.pth"
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=train_loss,
                metrics={},  # No metrics for now
                checkpoint_path=best_checkpoint_path
            )
            log.info(f"New best model saved with train_loss: {train_loss:.4f}")
    
    log.info("Training completed!")


if __name__ == "__main__":
    main()