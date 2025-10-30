from __future__ import annotations  # Enable postponed evaluation of annotations for forward references
import os
import sys
# Ensure repo root is on sys.path so `ulsd_model` can be imported without installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
    
import argparse  # Provide parsing utilities for command line arguments
import random  # Offer Python-level random number generation for reproducibility
from pathlib import Path  # Supply filesystem path manipulations with object-oriented interface
from typing import Any, Dict, Optional, Tuple  # Expose common typing aliases used throughout the script

import numpy as np  # Deliver numerical routines and deterministic seeding utilities
import torch  # Core PyTorch tensor library used for model training
import torch.nn.functional as F  # Include functional API for operations like one-hot and dropout
import torchvision.utils as vutils  # Provide helpers to save image grids for visualization
import yaml  # Handle configuration loading from YAML files
from torch import Tensor  # Clarify Tensor type for annotations without qualifying torch.Tensor
from torch.optim import Adam  # Import the Adam optimizer for training the UNet
from torch.utils.data import DataLoader  # Enable batched dataset loading
from tqdm import tqdm  # Supply progress bars for training loops

from con_stable_diff_model.datasets.MnistDatasets import MNISTDataset  # Custom MNIST dataset wrapper with conditioning metadata
from con_stable_diff_model.models.LinearNoiseScheduler import LinearNoiseScheduler  # Diffusion noise scheduler implementation
from con_stable_diff_model.models.VQVAE import VectorQuantizedVAE as VQVAE  # Pretrained VQ-VAE for encoding/decoding images
from con_stable_diff_model.modules.UNet import UNet  # Conditional UNet backbone used as denoiser
from con_stable_diff_model.utils.config import (  # Configuration helpers to read conditional settings safely
    get_config_value,  # Utility to fetch values with defaults from config dictionaries
    validate_class_config,  # Runtime validation ensuring class conditioning settings are complete
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose GPU when available, otherwise fall back to CPU


def load_config(path: Path) -> Dict[str, Any]:  # Load YAML configuration from disk into a dictionary
    """Read YAML configuration from disk."""  # Document function intent for clarity
    if not path.exists():  # Guard against missing configuration files
        raise FileNotFoundError(f"Config file not found: {path}")  # Emit helpful error message when path is invalid
    with path.open("r", encoding="utf-8") as handle:  # Open config file with explicit UTF-8 encoding
        return yaml.safe_load(handle)  # Parse YAML contents into native Python structures


def set_seed(seed: int) -> None:  # Make training deterministic given a provided seed
    """Seed Python, NumPy, and Torch RNGs for reproducibility."""  # Explain reproducibility purpose
    random.seed(seed)  # Seed Python standard library RNG
    np.random.seed(seed)  # Seed NumPy RNG for array operations
    torch.manual_seed(seed)  # Seed PyTorch CPU RNG
    if DEVICE.type == "cuda":  # Apply CUDA-specific seeding when GPU is active
        torch.cuda.manual_seed_all(seed)  # Seed all CUDA devices for consistent behaviour
    torch.backends.cudnn.deterministic = True  # Force deterministic CuDNN operations
    torch.backends.cudnn.benchmark = False  # Disable CuDNN auto-tuner to avoid nondeterminism


def ensure_dir(path: Path) -> None:  # Create directories if they are missing
    """Create a directory tree if it does not already exist."""  # Clarify helper behaviour
    path.mkdir(parents=True, exist_ok=True)  # Recursively create directories, ignoring if already present


def save_image_grid(images: Tensor, directory: Path, step: int, nrow: int, prefix: str) -> None:  # Persist qualitative image grids
    """Write a tiled grid of images to disk for qualitative inspection."""  # Describe visualization utility
    ensure_dir(directory)  # Guarantee that the destination directory exists
    grid = vutils.make_grid(  # Assemble a grid of images for viewing
        images.clamp(0.0, 1.0),  # Clamp tensor values to valid image range
        nrow=max(1, min(nrow, images.size(0))),  # Cap row count so it never exceeds batch size
        padding=2,  # Add spacing between images in the grid
    )
    vutils.save_image(grid, directory / f"{prefix}_{step:06d}.png")  # Write the grid to disk with step-indexed filename


def decode_latents_to_images(latents: Tensor, vqvae: VQVAE) -> Tensor:  # Convert latent tensors back into image space
    """Decode latent tensors back to image space and normalize to [0, 1]."""  # Summarize decoding behaviour
    with torch.no_grad():  # Disable gradients for evaluation-only decoding
        decoded = vqvae.decode(latents)  # Map latent codes through the VQ-VAE decoder
    decoded = decoded.clamp(-1.0, 1.0)  # Clip decoded values to the training range
    return ((decoded + 1.0) / 2.0).detach().cpu()  # Rescale to [0, 1], detach from graph, and move to CPU


def sample_diffusion_latents(  # Perform reverse diffusion sampling to generate latents
    model: UNet,  # Conditional UNet denoiser used for reverse process
    scheduler: LinearNoiseScheduler,  # Linear noise scheduler controlling diffusion steps
    num_samples: int,  # Number of latent samples to generate
    latent_shape: Tuple[int, ...],  # Shape of latent tensors (C, H, W)
    cond_input: Optional[Dict[str, Tensor]] = None,  # Optional conditioning dictionary for guided sampling
) -> Tensor:
    """Run the reverse DDPM process to draw latent samples."""  # Provide summary for function purpose
    was_training = model.training  # Remember training state so it can be restored later
    model.eval()  # Switch model to eval mode for deterministic behaviour

    with torch.no_grad():  # Disable autograd during sampling
        latents = torch.randn((num_samples, *latent_shape), device=DEVICE)  # Initialize latents with Gaussian noise
        predicted_x0: Optional[Tensor] = None  # Track the most recent reconstructed clean sample
        for timestep in range(scheduler.num_timesteps - 1, -1, -1):  # Iterate timesteps backwards through diffusion process
            t_tensor = torch.full((num_samples,), timestep, device=DEVICE, dtype=torch.long)  # Create tensor filled with current timestep index
            noise_pred = model(latents, t_tensor, cond_input=cond_input)  # Predict noise using the UNet conditioned on timestep (and optional class)
            latents, predicted_x0 = scheduler.sample_prev_timestep(latents, noise_pred, timestep)  # Step to previous timestep using scheduler equations
        samples = predicted_x0 if predicted_x0 is not None else latents  # Prefer final predicted clean samples when available

    if was_training:  # Restore model training state if it was originally training
        model.train()  # Re-enable training mode to keep dropout/batchnorm states consistent

    return samples.detach()  # Return detached samples to avoid gradient tracking downstream


def prepare_class_condition(  # Convert integer labels into one-hot conditioning with optional dropout
    class_indices: Tensor,  # Tensor of class indices supplied by the dataset
    condition_config: Dict[str, Any],  # Configuration chunk describing class conditioning behaviour
) -> Tensor:
    """Convert integer class targets to (optionally dropped) one-hot embeddings."""  # Clarify transformation performed
    class_cfg = condition_config["class_condition_config"]  # Extract class conditioning parameters
    num_classes = class_cfg["num_classes"]  # Determine total number of classes
    drop_prob = float(get_config_value(class_cfg, "cond_drop_prob", 0.0))  # Pull optional dropout probability for classifier-free guidance

    class_indices = class_indices.to(DEVICE)  # Move incoming class indices onto active device
    class_tensor = F.one_hot(class_indices, num_classes=num_classes).float()  # Create one-hot encodings for each index

    if drop_prob > 0.0:  # Apply dropout when probability is non-zero
        drop_mask = (torch.rand(class_tensor.size(0), device=class_tensor.device) < drop_prob).unsqueeze(-1)  # Sample Bernoulli mask per example
        class_tensor = torch.where(drop_mask, torch.zeros_like(class_tensor), class_tensor)  # Replace dropped embeddings with zeros

    return class_tensor  # Return final conditioning tensor for integration into timestep embedding


def build_dataset(  # Instantiate dataset based on configuration settings
    dataset_name: str,  # Name indicating which dataset to load
    dataset_cfg: Dict[str, Any],  # Dataset configuration dictionary
    condition_config: Optional[Dict[str, Any]],  # Conditioning configuration passed down for dataset packaging
) -> MNISTDataset:
    """Instantiate the dataset specified by the configuration."""  # Describe the factory function
    if dataset_name != "mnist":  # Restrict training script to supported dataset
        raise ValueError(f"Unsupported dataset '{dataset_name}'.")  # Raise clear error for unsupported options
    return MNISTDataset(  # Construct dataset with provided arguments
        dataset_split="train",  # Load training partition
        data_root=dataset_cfg["im_path"],  # Point to directory containing images
        condition_config=condition_config,  # Pass conditioning metadata so dataset can emit labels when needed
    )


def train(config: Dict[str, Any]) -> None:  # Execute the DDPM training loop using supplied configuration
    """Train a conditional DDPM on VQ-VAE latents."""  # Provide overview of training routine
    diffusion_cfg = config["diffusion_params"]  # Extract diffusion hyperparameters controlling noise schedule
    dataset_cfg = config["dataset_params"]  # Retrieve dataset configuration for data loading
    unet_cfg = config["UnetParams"]  # Obtain UNet architectural parameters
    vqvae_cfg = config["VQVAE"]  # Load VQ-VAE architectural settings
    train_cfg = config["train_params"]  # Pull high-level training hyperparameters
    diffusion_model_cfg = config.get("cond_model", {})  # Optionally grab conditional model configuration block

    condition_config = get_config_value(diffusion_model_cfg, "condition_config", None)  # Extract conditioning options if present
    condition_types = condition_config["condition_types"] if condition_config else []  # Enumerate enabled conditioning modalities
    if "class" in condition_types:  # Validate class conditioning when requested
        validate_class_config(condition_config)  # Ensure required keys like num_classes exist

    set_seed(train_cfg.get("seed", 0))  # Seed all RNGs for reproducibility using configured seed

    dataset = build_dataset(dataset_cfg["name"], dataset_cfg, condition_config)  # Instantiate dataset according to configuration
    data_loader = DataLoader(  # Wrap dataset with DataLoader for batching
        dataset,  # Dataset instance providing tensors (and optional conditioning dict)
        batch_size=train_cfg["ldm_batch_size"],  # Number of samples per batch
        shuffle=True,  # Shuffle data each epoch to improve training stability
        num_workers=train_cfg.get("num_workers", 0),  # Optional background workers for data loading
        pin_memory=DEVICE.type == "cuda",  # Pin host memory to speed up GPU transfers
    )

    output_dir = Path(train_cfg.get("task_name_ddpm_cond", train_cfg["task_name"]))  # Directory to store checkpoints and samples
    vqvae_dir = Path(train_cfg["task_name"])  # Directory holding VQ-VAE checkpoints and latents
    ensure_dir(output_dir)  # Create output directory if it does not exist

    scheduler = LinearNoiseScheduler(  # Instantiate diffusion noise scheduler using config parameters
        device=DEVICE,  # Ensure scheduler tensors reside on training device
        num_timesteps=diffusion_cfg["num_timesteps"],  # Total diffusion steps to traverse
        beta_start=diffusion_cfg["beta_start"],  # Starting beta value for linear schedule
        beta_end=diffusion_cfg["beta_end"],  # Ending beta value for schedule
    )

    unet_params = dict(unet_cfg)  # Clone UNet config to avoid mutating original dictionary
    unet_params["im_channels"] = vqvae_cfg["z_channels"]  # Override input channels to match encoded latent channels
    if diffusion_model_cfg:  # Attach conditional model configuration when available
        unet_params["model_config"] = diffusion_model_cfg  # Pass conditioning configuration into UNet constructor
    model = UNet(UnetParams=unet_params).to(DEVICE)  # Build UNet with provided parameters and move to training device
    model.train()  # Set UNet to training mode so layers like dropout operate correctly

    vqvae = VQVAE(  # Instantiate VQ-VAE used to convert images to latent space
        input_channels=dataset_cfg["im_channels"],  # Match dataset image channel count (e.g., 1 for MNIST)
        VQVAE=vqvae_cfg,  # Provide VQ-VAE architectural configuration
    ).to(DEVICE)  # Move VQ-VAE to device for encoding operations
    vqvae.eval()  # Switch VQ-VAE to evaluation mode to freeze statistics
    for param in vqvae.parameters():  # Iterate over VQ-VAE parameters
        param.requires_grad = False  # Disable gradients to keep VQ-VAE frozen during diffusion training

    vqvae_ckpt = vqvae_dir / train_cfg["vqvae_autoencoder_ckpt_name"]  # Construct path to VQ-VAE checkpoint
    if not vqvae_ckpt.exists():  # Ensure pretrained checkpoint is available
        raise FileNotFoundError(  # Raise error when checkpoint missing to inform user
            f"VQ-VAE checkpoint not found at {vqvae_ckpt}. Train the autoencoder before launching DDPM training."
        )
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=DEVICE))  # Load pretrained VQ-VAE weights onto device

    optimizer = Adam(model.parameters(), lr=train_cfg["ldm_lr"])  # Configure Adam optimizer using learning rate from config
    criterion = torch.nn.MSELoss()  # Choose mean squared error loss for noise prediction objective
    num_epochs = train_cfg["ldm_epochs"]  # Determine number of epochs to train

    samples_root = output_dir / "ddpm_samples"  # Base directory for storing generated sample grids
    recon_dir = samples_root / "recon"  # Directory for reconstruction visualizations
    sample_dir = samples_root / "samples"  # Directory for pure diffusion samples
    ensure_dir(recon_dir)  # Create reconstruction directory if missing
    ensure_dir(sample_dir)  # Create sampling directory if missing

    save_every = max(1, int(train_cfg.get("ldm_img_save_steps", 1000)))  # Determine interval for saving sample grids
    viz_samples = max(1, int(train_cfg.get("num_samples", 16)))  # Number of samples to visualize per save event
    viz_rows = max(1, int(train_cfg.get("num_grid_rows", int(np.ceil(np.sqrt(viz_samples))))))  # Row count for image grid layout

    global_step = 0  # Initialize counter tracking total gradient updates
    sample_index = 0  # Track index for saved visualization batches
    latent_shape: Optional[Tuple[int, ...]] = None  # Record latent tensor shape once determined

    for epoch in range(num_epochs):  # Iterate through each training epoch
        epoch_losses: list[float] = []  # Collect per-batch losses for averaging
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):  # Loop over batches with progress feedback
            if condition_config:  # Handle datasets that return conditioning information
                images, raw_cond = batch  # Unpack image tensor and conditioning dict
            else:
                images = batch  # Receive only images when no conditioning
                raw_cond = {}  # Use empty dictionary for consistent handling

            images = images.to(DEVICE, non_blocking=True).float()  # Move images to device and ensure float dtype

            optimizer.zero_grad(set_to_none=True)  # Reset gradients efficiently before forward pass

            with torch.no_grad():  # Freeze gradients while encoding with VQ-VAE
                latents, _ = vqvae.encode(images)  # Encode images into latent space (discarding quantization losses)
            latents = latents.detach()  # Detach latents from computation graph

            if latent_shape is None:  # Record latent tensor shape once for sampling routine
                latent_shape = tuple(latents.shape[1:])  # Capture channel and spatial dimensions

            cond_input: Optional[Dict[str, Tensor]] = None  # Initialize conditioning dictionary passed to UNet
            if condition_config and raw_cond:  # Only process conditioning when provided
                cond_input = {}  # Create dictionary to hold conditioning tensors
                if "class" in condition_types:  # Convert class labels into embeddings when enabled
                    class_indices = raw_cond["class"].to(DEVICE)  # Move label indices to current device
                    cond_input["class"] = prepare_class_condition(class_indices, condition_config)  # Populate class conditioning entry

            noise = torch.randn_like(latents)  # Sample Gaussian noise matching latent shape
            timesteps = torch.randint(  # Draw random diffusion timesteps per sample
                low=0,
                high=diffusion_cfg["num_timesteps"],
                size=(latents.shape[0],),
                device=latents.device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)  # Corrupt clean latents according to sampled timesteps

            noise_pred = model(noisy_latents, timesteps, cond_input=cond_input)  # Predict noise using conditional UNet
            loss = criterion(noise_pred, noise)  # Compute denoising objective as MSE between predicted and target noise
            loss.backward()  # Backpropagate gradients through UNet
            optimizer.step()  # Update UNet parameters with optimizer

            epoch_losses.append(loss.item())  # Store loss scalar for epoch statistics
            global_step += 1  # Increment global optimization step counter

            if global_step % save_every == 0 and latent_shape is not None:  # Periodically save reconstructions and samples
                real_count = min(viz_samples, latents.size(0))  # Determine number of real examples to visualize
                real_latents = latents[:real_count].detach()  # Slice latent batch for visualization
                real_images = decode_latents_to_images(real_latents, vqvae)  # Decode latents to image space for reconstructions

                sampled_latents = sample_diffusion_latents(  # Generate samples via reverse diffusion
                    model=model,
                    scheduler=scheduler,
                    num_samples=viz_samples,
                    latent_shape=latent_shape,
                    cond_input=None,
                )
                sampled_images = decode_latents_to_images(sampled_latents, vqvae)  # Decode generated latents to image space

                save_image_grid(real_images, recon_dir, sample_index, viz_rows, "recon")  # Save reconstruction grid to disk
                save_image_grid(sampled_images, sample_dir, sample_index, viz_rows, "sample")  # Save generated sample grid to disk
                tqdm.write(f"Saved diffusion samples at step {global_step} (index {sample_index}).")  # Log sampling event to console
                sample_index += 1  # Increment sample index for future saves

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0  # Compute mean loss over the epoch
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss {mean_loss:.4f}")  # Report epoch progress and loss

        torch.save(  # Persist model checkpoint after each epoch
            model.state_dict(),
            output_dir / train_cfg["ldm_ckpt_name"],
        )

    print("Done Training...")  # Indicate completion of training procedure


def parse_args() -> argparse.Namespace:  # Parse command-line options for this script
    parser = argparse.ArgumentParser(description="Arguments for conditional DDPM training")  # Describe script purpose in CLI help
    parser.add_argument(  # Register --config CLI flag
        "--config",
        dest="config_path",
        default="./con_stable_diff_model/config.yml",
        type=str,
    )
    return parser.parse_args()  # Execute parsing and return populated namespace


def main() -> None:  # Entry point when executing script directly
    args = parse_args()  # Parse CLI arguments to retrieve configuration path
    config = load_config(Path(args.config_path))  # Load training configuration from YAML file
    train(config)  # Launch DDPM training using loaded configuration


if __name__ == "__main__":  # Ensure script runs only when executed directly, not imported
    main()  # Invoke main function to start program
