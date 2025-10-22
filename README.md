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

## Development Tips
- Keep configuration dictionaries in sync with the validation helpers to avoid runtime assertions.
- When introducing new conditioning modalities, extend the helper functions in `utils/config.py` and follow the commenting pattern used in `modules/UNet.py`.
