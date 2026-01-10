# Conditional-Latent-Stable-Diffusion-Models

## Overview
This repository explores conditional latent diffusion pipelines that pair a variational autoencoder (or vector-quantised autoencoder) with a UNet denoiser. The UNet operates in latent space and now ships with flexible conditioning hooks so it can ingest labels, text embeddings, or auxiliary images when guiding the denoising trajectory.

## Key Features
- **Class-aware UNet** – The denoiser accepts class logits/one-hot vectors and projects them into the timestep embedding, enabling classifier-free guidance or hard class conditioning.
- **Extensible conditioning utilities** – `con_stable_diff_model/utils/config.py` centralises validation helpers for optional conditioning blocks, simplifying configuration management.
- **Modular UNet blocks** – Encoder, bottleneck, and decoder stages are composed from reusable residual/attention blocks defined in `con_stable_diff_model/models/UNetBlocks.py`.

## Configuration
Edit the config file you plan to train with (for example `con_stable_diff_model/config_celeb.yml` or `con_stable_diff_model/config_celeb_text_image.yml`). Within the `cond_model` or `UnetParams` sections you can enable conditioning by adding a `condition_config` block, for example:

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
- **Text conditioning** – Use `text_condition_config` to pick the encoder (`clip` or `bert`) and match `text_embed_dim` to the model output (CLIP ViT-B/16 -> 512, DistilBERT -> 768).

### Conditioning Implementation Details
Below is how each conditioning signal is injected into the UNet. Shapes are shown for a batch size `B`.

```
Inputs
  image x0 (B, 3, H, W) --VQ-VAE--> latents z (B, C, h, w)
  timestep t ---------------------> time_emb (B, T)
  class idx ----------------------> one_hot (B, num_classes)
  text prompt --------------------> text_embed (B, L, D)
  image mask ---------------------> mask (B, Cmask, Hm, Wm)

UNet Conditioning Flow
  time_emb --MLP--> t_proj (B, T)
  class one_hot --Embedding--> class_emb (B, T)
  t_proj + class_emb -------------------------> used in all ResNet blocks

  mask --1x1 conv--> mask_proj (B, Cimg, h, w)
  concat([z, mask_proj], dim=1) -> (B, C + Cimg, h, w) -> conv_in

  text_embed (B, L, D) --> cross-attn context in Down/Mid/Up blocks
```

**Tensor stacking rules**
- **Image conditioning**: stack by concatenating along channels:  
  `z: (B, C, h, w)` + `mask_proj: (B, Cimg, h, w)` → `(B, C + Cimg, h, w)`.
- **Class conditioning**: no stacking; class embedding is **added** to the timestep embedding (`t_proj + class_emb`).
- **Text conditioning**: no stacking; text embedding is passed as **context** to cross-attention layers.

**Classifier-free guidance (dropout)**
- `cond_drop_prob` can drop class, image, or text conditions during training.
- Dropped class embeddings become zeros, dropped image conditioning becomes zeros, and dropped text embeddings use an empty prompt embedding.

### Conditional Options (config)
You can enable one or more modalities with `condition_types`. Common combinations:
- `['class']` for label guidance.
- `['image']` for mask/segmentation guidance.
- `['text']` for prompt guidance.
- `['text', 'image']` for hybrid prompt + mask guidance.

Minimal text+image configuration example:
```yaml
cond_model:
  condition_config:
    condition_types: ['text', 'image']
    text_condition_config:
      text_embed_model: 'clip'
      text_embed_dim: 512
      cond_drop_prob: 0.1
    image_condition_config:
      image_condition_input_channels: 18
      image_condition_output_channels: 3
      image_condition_h: 256
      image_condition_w: 256
      cond_drop_prob: 0.1
```

If you use text conditioning, make sure the dataset supplies a `text` field (CelebA-HQ prompts or captions are handled in `con_stable_diff_model/datasets/CelebDataset.py`).

## Class Conditioning Details
- Enable label guidance by setting `condition_types: ['class']` in `config.yml` and making sure the dataset emits class indices (see `MNISTDataset` for an example).
- During training the helper `prepare_class_condition` converts integer labels to one-hot encodings and applies optional classifier-free guidance dropout controlled by `cond_drop_prob`.
- The UNet validates that a class tensor is always present when class conditioning is active; this prevents mismatches between configuration and runtime inputs.
- Sampling can be steered toward specific digits by creating a one-hot tensor of the desired classes and passing it as `cond_input` when calling `sample_diffusion_latents`.
- Adjust `num_classes` to match your dataset and consider increasing `cond_drop_prob` if you want stronger classifier-free guidance during inference.

## Development Tips
- Keep configuration dictionaries in sync with the validation helpers to avoid runtime assertions.
- When introducing new conditioning modalities, extend the helper functions in `utils/config.py` and follow the commenting pattern used in `modules/UNet.py`.

## Training Overview
The repo exposes three training stages that build on each other:

| Stage | Script / `main.py --mode` | Config example | Purpose |
|-------|---------------------------|----------------|---------|
| VQ-VAE autoencoder | `scripts/train_vqvae.py` / `python main.py --mode vqvae --config con_stable_diff_model/config_vqvae_celeb.yml` | `config_vqvae_celeb.yml` | Learns an image-to-latent encoder/decoder for CelebA-HQ or MNIST. Produces `vqvae_autoencoder_ckpt.pth`. |
| Latent DDPM (unconditional)| `scripts/train_ddpm_vqvae.py` / `python main.py --mode ddpm_vqvae --config ...` | `config_vqvae_celeb.yml` | Trains a DDPM directly on latents produced by the VQ-VAE. Saves `celeb_ddpm_vqvae/ddpm_ckpt.pth`. |
| Conditional DDPM | `scripts/train_ddpm_cond.py` / `python main.py --mode ddpm_cond --config con_stable_diff_model/config_celeb.yml` | `config_celeb.yml` | Adds class/image/text conditioning on top of the latent DDPM for guided synthesis. |
| Conditional DDPM (text + image) | `scripts/train_ddpm_image_text_cond.py` / `python main.py --mode ddpm_image_text_cond --config con_stable_diff_model/config_celeb_text_image.yml` | `config_celeb_text_image.yml` | Hybrid prompt + mask conditioning with side-by-side visualization outputs. |

Each config follows the same structure:

```yaml
VQVAE:          # Autoencoder architecture (channels, attention, codebook)
cond_model:     # Optional extra conditioning blocks specific to UNet
dataset_params: # Dataset name (mnist, celeb), paths, image size/channels
diffusion_params: # Beta schedule for DDPM
UnetParams:     # Base UNet architecture; `im_channels` must match VQ-VAE z_channels
train_params:   # Seeds, batch sizes, lr, checkpoint names, output folders
```

Key `train_params` entries:
- `task_name`: root folder containing autoencoder checkpoints and latents (e.g., `celeb/`).
- `task_name_ddpm_vqvae` / `task_name_ddpm_cond`: output folders for the DDPM trainers.
- `vqvae_autoencoder_ckpt_name`: file inside `task_name` holding the VQ-VAE weights.
- `vqvae_latent_dir_name`: optional cache of encoded latents; if absent the DDPM trainer encodes on the fly.
- `autoencoder_*` keys control VQ-VAE optimisation (batch size, lr, gradient accumulation, logging cadence).
- `ldm_*` keys control the DDPM stages.

### 1. Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pyyaml tqdm pillow
```

### 2. Prepare datasets
- **MNIST**: directory structure `{root}/{digit}/{image.png}`. Set `dataset_params.name: mnist`, `im_channels: 1`, `im_size: 28`.
- **CelebA-HQ**: set `dataset_params.name: celeb` (or `celebhq`), point `im_path` to the extracted CelebAMask-HQ root, and ensure masks exist if you use image conditioning. `dataset_factory` automatically selects `CelebDataset` for these names.

### 3. Train the VQ-VAE
```bash
python main.py --mode vqvae --config con_stable_diff_model/config_vqvae_celeb.yml
```
This writes `celeb/vqvae_autoencoder_ckpt.pth`, discriminator checkpoints, and reconstruction grids under `celeb/vqvae_autoencoder_samples/`.

### 4. Optional: cache latents
After the VQ-VAE is trained you can precompute latents if desired (set `train_params.save_latents: True` in the VQ-VAE config or adapt your workflow). When absent, the DDPM trainer encodes inputs on the fly, which is slower but functional.

### 5. Train the latent DDPM
```bash
python main.py --mode ddpm_vqvae --config con_stable_diff_model/config_vqvae_celeb.yml
```
This loads the VQ-VAE checkpoint, consumes `dataset_params`, and saves DDPM weights plus samples to `celeb_ddpm_vqvae/`.

### 6. Train the conditional DDPM
```bash
python main.py --mode ddpm_cond --config con_stable_diff_model/config_celeb.yml
```
Make sure `train_params.task_name` in `config_celeb.yml` points to the folder containing the autoencoder checkpoint from step 3, and `cond_model.condition_config` matches your conditioning modality (e.g., image masks with 18 channels for CelebAMask-HQ). Outputs land in `celeb_ddpm_vqvae_cond/`.

### 7. Train text + image conditional DDPM
```bash
python main.py --mode ddpm_image_text_cond --config con_stable_diff_model/config_celeb_text_image.yml
```
This configuration enables `['text', 'image']` conditioning. Samples are saved under `.../<task_name>_samples/` with:
- `recon/`: autoencoder reconstructions.
- `samples/`: diffusion samples.
- `viz/`: side-by-side grids showing image condition, text prompt, reconstruction, noisy decode, and denoised decode.

### 7. Outputs and checkpoints
- VQ-VAE: `celeb/vqvae_autoencoder_ckpt.pth`, `celeb/vqvae_discriminator_ckpt.pth`, sample grids under `vqvae_autoencoder_samples/`.
- Latent DDPM: `celeb_ddpm_vqvae/ddpm_ckpt.pth`, sample grids under `ddpm_samples/`.
- Conditional DDPM: `celeb_ddpm_vqvae_cond/ddpm_ckpt.pth`, conditioned vs. reconstruction grids under `ddpm_samples/{recon,samples}/`.

To resume training place the relevant checkpoint back in the expected folder and rerun the same command; the script overwrites the checkpoint after the next epoch.

## Related Resources
- To train the companion `ddpm_vqvae` pipeline, refer to the upstream project at [`StableDiffusion-ULDM`](https://github.com/dnjegovanovic/StableDiffusion-ULDM) for setup instructions and pretrained assets.
