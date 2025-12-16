import os

import pytest

from con_stable_diff_model.datasets.CelebDataset import CelebDataset


def _find_dataset_root():
    """
    Resolve a CelebAMask-HQ root by checking env var first, then a common default.
    """
    candidates = [
        os.environ.get("CELEBAMASK_HQ_ROOT"),
        "/home/dusan/Desktop/ML_PROJECTS/ml_datasets/CelebAMask-HQ",
    ]
    for root in candidates:
        if root and os.path.isfile(os.path.join(root, "CelebAMask-HQ-attribute-anno.txt")):
            return root
    return None


def _build_condition_config():
    return {
        "condition_types": ["text"],
        "text_condition_config": {
            "text_embed_model": "clip",
            "train_text_embed_model": False,
            "text_embed_dim": 256,
            "cond_drop_prob": 0.1,
        },
    }


@pytest.mark.slow
def test_celeb_attribute_prompts_load():
    """
    Ensure every image gets at least one synthesized text prompt from attributes.
    Prints a few filename->prompt pairs for manual inspection.
    """
    root = _find_dataset_root()
    if root is None:
        pytest.skip("CelebAMask-HQ root not available (set CELEBAMASK_HQ_ROOT to override).")

    dataset = CelebDataset(
        split="train",
        im_path=root,
        im_size=256,
        im_channels=3,
        condition_config=_build_condition_config(),
    )

    assert len(dataset) > 0, "Dataset should not be empty"

    missing = [idx for idx, texts in enumerate(dataset.texts) if len(texts) == 0]
    assert not missing, f"Missing text prompts for {len(missing)} images (first few: {missing[:5]})"

    num_to_show = min(3, len(dataset))
    print("Sample filename -> prompt pairs:")
    for i in range(num_to_show):
        fname = os.path.basename(dataset.images[i])
        prompt = dataset.texts[i][0] if dataset.texts[i] else "<no text>"
        print(f"[{i}] {fname} -> {prompt}")

    # Sanity check that we produced prompts
    assert len(dataset.texts[0]) > 0
