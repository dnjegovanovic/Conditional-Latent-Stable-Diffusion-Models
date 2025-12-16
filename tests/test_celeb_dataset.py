from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from con_stable_diff_model.datasets.CelebDataset import CelebDataset

DATA_ROOT = Path("/home/dusan/Desktop/ML_PROJECTS/ml_datasets/CelebAMask-HQ")
CONDITION_CONFIG = {
    "condition_types": ["image"],
    "image_condition_config": {
        "image_condition_input_channels": 18,
        "image_condition_output_channels": 3,
        "image_condition_h": 256,
        "image_condition_w": 256,
        "cond_drop_prob": 0.1,
    },
}

requires_real_data = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason=f"CelebAMask-HQ dataset not found at {DATA_ROOT}",
)


@requires_real_data
def test_celeb_dataset_returns_normalized_tensor():
    dataset = CelebDataset(
        split="train",
        im_path=str(DATA_ROOT),
        im_size=128,
        im_channels=3,
    )

    assert len(dataset) > 0

    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.shape == (3, 128, 128)
    assert sample.min().item() >= -1.05
    assert sample.max().item() <= 1.05


@requires_real_data
def test_celeb_dataset_falls_back_when_latents_missing():
    dataset = CelebDataset(
        split="train",
        im_path=str(DATA_ROOT),
        im_size=64,
        im_channels=3,
        use_latents=True,
        latent_path=str(DATA_ROOT / "non_existent_latents"),
    )

    assert dataset.use_latents is False

    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)


@requires_real_data
def test_celeb_dataset_returns_mask_tensor_when_conditioned_on_image():
    dataset = CelebDataset(
        split="train",
        im_path=str(DATA_ROOT),
        im_size=64,
        im_channels=3,
        condition_config=CONDITION_CONFIG,
    )

    image_tensor, cond_inputs = dataset[0]

    assert isinstance(image_tensor, torch.Tensor)
    assert isinstance(cond_inputs, dict)
    assert "image" in cond_inputs

    mask_tensor = cond_inputs["image"]
    assert isinstance(mask_tensor, torch.Tensor)
    assert mask_tensor.dtype == torch.float32
    assert mask_tensor.shape == (
        CONDITION_CONFIG["image_condition_config"]["image_condition_input_channels"],
        CONDITION_CONFIG["image_condition_config"]["image_condition_h"],
        CONDITION_CONFIG["image_condition_config"]["image_condition_w"],
    )


def test_mask_channels_follow_annotation_files(tmp_path):
    """
    Build a tiny synthetic CelebAMask layout and ensure the dataset places the
    correct binary masks into the expected semantic channels, while leaving
    channels without annotations empty.
    """
    data_root = tmp_path / "celeb_synth"
    images_dir = data_root / "CelebA-HQ-img"
    mask_dir = data_root / "CelebAMask-HQ-mask-anno" / "0"
    images_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    # Write a single 2x2 mask for two classes with different patterns.
    Image.new("RGB", (8, 8), color=(10, 20, 30)).save(images_dir / "00000.png")
    Image.fromarray(np.array([[0, 255], [0, 255]], dtype=np.uint8)).save(
        mask_dir / "00000_skin.png"
    )
    Image.fromarray(np.array([[255, 0], [255, 0]], dtype=np.uint8)).save(
        mask_dir / "00000_hair.png"
    )

    condition_config = {
        "condition_types": ["image"],
        "image_condition_config": {
            "image_condition_input_channels": 18,
            "image_condition_output_channels": 3,
            "image_condition_h": 4,
            "image_condition_w": 4,
            "cond_drop_prob": 0.1,
        },
    }

    dataset = CelebDataset(
        split="train",
        im_path=str(data_root),
        im_size=8,
        im_channels=3,
        condition_config=condition_config,
    )

    _, cond_inputs = dataset[0]
    mask = cond_inputs["image"]

    skin_idx = dataset.cls_to_idx_map["skin"]
    hair_idx = dataset.cls_to_idx_map["hair"]
    nose_idx = dataset.cls_to_idx_map["nose"]  # No mask written for "nose"

    assert mask.shape == (
        condition_config["image_condition_config"]["image_condition_input_channels"],
        condition_config["image_condition_config"]["image_condition_h"],
        condition_config["image_condition_config"]["image_condition_w"],
    )
    assert mask[skin_idx].sum().item() > 0  # picks up 00000_skin.png
    assert mask[hair_idx].sum().item() > 0  # picks up 00000_hair.png
    assert mask[nose_idx].sum().item() == 0  # empty channel stays zero
