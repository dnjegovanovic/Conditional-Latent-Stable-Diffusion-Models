from pathlib import Path

import pytest
import torch

from con_stable_diff_model.datasets.CelebDataset import CelebDataset

DATA_ROOT = Path("/home/dusan/Desktop/ML_PROJECTS/ml_datasets/CelebAMask-HQ")

pytestmark = pytest.mark.skipif(
    not DATA_ROOT.exists(),
    reason=f"CelebAMask-HQ dataset not found at {DATA_ROOT}",
)


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
