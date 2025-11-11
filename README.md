# Conditional-Latent-Stable-Diffusion-Models

## Overview
This repository explores conditional latent diffusion pipelines that pair a variational autoencoder (or vector-quantised autoencoder) with a UNet denoiser. The UNet operates in latent space and now ships with flexible conditioning hooks so it can ingest labels, text embeddings, or auxiliary images when guiding the denoising trajectory.

## Key Features
- **Class-aware UNet** – The denoiser accepts class logits/one-hot vectors and projects them into the timestep embedding, enabling classifier-free guidance or hard class conditioning.
- **Extensible conditioning utilities** – `con_stable_diff_model/utils/config.py` centralises validation helpers for optional conditioning blocks, simplifying configuration management.
- **Modular UNet blocks** – Encoder, bottleneck, and decoder stages are composed from reusable residual/attention blocks defined in `con_stable_diff_model/models/UNetBlocks.py`.

## Configuration
Edit `con_stable_diff_model/config.yml` to adjust architecture or conditioning settings. Within the `cond_model` or `UnetParams` sections you can enable conditioning by adding a `condition_config` block, for example:

```yaml
condition_config:
  condition_types: ['class']
  class_condition_config:
    num_classes: 10
    cond_drop_prob: 0.1
```

## Conditioning Inputs
- **Class conditioning** – Pass `cond_input={"class": class_tensor}` where `class_tensor` has shape `(batch_size, num_classes)` and holds probabilities or one-hot encodings.
- **Image conditioning** – If configured, supply `cond_input["image"]` with the same spatial resolution as the latent being denoised; it will be projected and concatenated channel-wise.
- **Text conditioning** – Reserved for future text encoders; set `text_condition_config` to describe the embedding dimensionality.

## Class Conditioning Details
- Enable label guidance by setting `condition_types: ['class']` in `config.yml` and making sure the dataset emits class indices (see `MNISTDataset` for an example).
- During training the helper `prepare_class_condition` converts integer labels to one-hot encodings and applies optional classifier-free guidance dropout controlled by `cond_drop_prob`.
- The UNet validates that a class tensor is always present when class conditioning is active; this prevents mismatches between configuration and runtime inputs.
- Sampling can be steered toward specific digits by creating a one-hot tensor of the desired classes and passing it as `cond_input` when calling `sample_diffusion_latents`.
- Adjust `num_classes` to match your dataset and consider increasing `cond_drop_prob` if you want stronger classifier-free guidance during inference.

## Development Tips
- Keep configuration dictionaries in sync with the validation helpers to avoid runtime assertions.
- When introducing new conditioning modalities, extend the helper functions in `utils/config.py` and follow the commenting pattern used in `modules/UNet.py`.

## Training
1. **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the wheel that matches your CUDA/CPU setup
   pip install numpy pyyaml tqdm pillow
   ```
   The training script relies only on the libraries above plus the standard library.

2. **Prepare data and the VQ-VAE checkpoint**
   - `dataset_params.im_path` must point to a directory laid out as `{root}/{digit}/{image.png}`. The default configuration expects an extracted MNIST dataset; each digit folder (0–9) contains its samples.
   - Train or download a VQ-VAE checkpoint first. The DDPM expects to find `train_params.vqvae_autoencoder_ckpt_name` (defaults to `vqvae_autoencoder_ckpt.pth`) inside the folder `train_params.task_name` (defaults to `mnist/`). If you use your own dataset, place the pretrained autoencoder weights in the analogous directory.

3. **Adjust `con_stable_diff_model/config.yml`**
   - Update `dataset_params` (paths, number of channels, size).
   - Set the conditioning you want under `cond_model.condition_config`.
   - Modify `train_params` for batch size, epochs, learning rates, output folders, etc.

4. **Launch DDPM training**
   ```bash
   python scripts/train_ddpm_cond.py --config con_stable_diff_model/config.yml
   # or equivalently
   python main.py --config con_stable_diff_model/config.yml
   ```
   The script will: load the frozen VQ-VAE, build the conditional UNet, train for `train_params.ldm_epochs`, and periodically write grids under `<task_name_ddpm_cond>/ddpm_samples/`.

5. **Outputs and checkpoints**
   - Checkpoints are saved each epoch to `<task_name_ddpm_cond>/<ldm_ckpt_name>` (default `mnist_ddpm_vqvae_cond/ddpm_ckpt.pth`).
   - Reconstructions are logged under `ddpm_samples/recon/` and unconditional/conditional generations under `ddpm_samples/samples/`.
   - To resume, keep the checkpoint in place and rerun the script with the same config; PyTorch will overwrite the file after the first subsequent epoch.

## Related Resources
- To train the companion `ddpm_vqvae` pipeline, refer to the upstream project at [`StableDiffusion-ULDM`](https://github.com/dnjegovanovic/StableDiffusion-ULDM) for setup instructions and pretrained assets.
