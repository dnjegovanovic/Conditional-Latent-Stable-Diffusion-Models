import glob  # Globs file paths based on wildcard patterns
import os  # OS-level utilities for path handling
import random  # Random selection for sampling captions
import torch  # Core tensor library for mask tensors
import torchvision  # Provides transforms for preprocessing images
import numpy as np  # Offers array utilities when processing segmentation masks
from PIL import Image  # Used to open image and mask files
from con_stable_diff_model.utils.diff_utils import (
    load_latents,
)  # Helper to load cached latent tensors
from tqdm import tqdm  # Progress bar for iterating over many files
from torch.utils.data.dataset import Dataset  # Base class for PyTorch datasets


class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(
        self,
        split,
        im_path,
        im_size=256,
        im_channels=3,
        im_ext="jpg",
        use_latents=False,
        latent_path=None,
        condition_config=None,
    ):
        self.split = split  # Track which split (train/val/test) is being used
        self.im_size = im_size  # Desired resolution after resizing/cropping
        self.im_channels = (
            im_channels  # Number of channels expected in the image tensor
        )
        self.im_ext = im_ext  # Default extension when searching for files
        self.im_path = im_path  # Root directory containing the dataset assets
        self.latent_maps = None  # Placeholder for optional latent representations
        self.use_latents = (
            False  # Flag to indicate whether to return latents instead of pixels
        )

        self.condition_types = (
            [] if condition_config is None else condition_config["condition_types"]
        )  # Enumerate enabled conditioning modalities

        self.idx_to_cls_map = {}  # Map from mask channel index to semantic label
        self.cls_to_idx_map = {}  # Reverse map from label string to channel index

        if "image" in self.condition_types:
            self.mask_channels = condition_config["image_condition_config"][
                "image_condition_input_channels"
            ]  # Number of one-hot channels in mask
            self.mask_h = condition_config["image_condition_config"][
                "image_condition_h"
            ]  # Target mask height
            self.mask_w = condition_config["image_condition_config"][
                "image_condition_w"
            ]  # Target mask width

        self.images, self.texts, self.masks = self.load_images(
            im_path
        )  # Collect file paths and ancillary metadata

        # Whether to load images or to load latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(
                latent_path
            )  # Load precomputed latents if requested
            if len(latent_maps) == len(self.images):
                self.use_latents = True  # Enable latent mode once counts align
                self.latent_maps = latent_maps  # Cache the latent dictionary
                print(
                    "Found {} latents".format(len(self.latent_maps))
                )  # Report number of latents discovered
            else:
                print(
                    "Latents not found"
                )  # Warn when no matching latents are available

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(
            im_path
        )  # Ensure dataset root exists
        ims = []  # Store image file paths
        fnames = glob.glob(
            os.path.join(im_path, "CelebA-HQ-img/*.{}".format("png"))
        )  # Gather PNG files
        fnames += glob.glob(
            os.path.join(im_path, "CelebA-HQ-img/*.{}".format("jpg"))
        )  # Gather JPG files
        fnames += glob.glob(
            os.path.join(im_path, "CelebA-HQ-img/*.{}".format("jpeg"))
        )  # Gather JPEG files
        texts = []  # Placeholder for caption lists
        masks = []  # Placeholder for mask file paths

        if "image" in self.condition_types:
            label_list = [
                "skin",
                "nose",
                "eye_g",
                "l_eye",
                "r_eye",
                "l_brow",
                "r_brow",
                "l_ear",
                "r_ear",
                "mouth",
                "u_lip",
                "l_lip",
                "hair",
                "hat",
                "ear_r",
                "neck_l",
                "neck",
                "cloth",
            ]  # Semantic labels provided by CelebAMask-HQ
            self.idx_to_cls_map = {
                idx: label_list[idx] for idx in range(len(label_list))
            }  # Map channel index to label name
            self.cls_to_idx_map = {
                label_list[idx]: idx for idx in range(len(label_list))
            }  # Map label name to channel index

        for fname in tqdm(fnames):
            ims.append(fname)  # Track every discovered image path

            if "text" in self.condition_types:
                im_name = os.path.split(fname)[1].split(".")[
                    0
                ]  # Derive base filename to look up caption file
                captions_im = []  # Collect captions for this image
                with open(
                    os.path.join(im_path, "celeba-caption/{}.txt".format(im_name))
                ) as f:
                    for line in f.readlines():
                        captions_im.append(
                            line.strip()
                        )  # Strip newline characters from caption strings
                texts.append(
                    captions_im
                )  # Append list of captions for the current image

            if "image" in self.condition_types:
                im_name = int(
                    os.path.split(fname)[1].split(".")[0]
                )  # Mask files use numeric naming
                masks.append(
                    os.path.join(
                        im_path, "CelebAMask-HQ-mask", "{}.png".format(im_name)
                    )
                )  # Derive mask path
        if "text" in self.condition_types:
            assert len(texts) == len(
                ims
            ), "Condition Type Text but could not find captions for all images"  # Validate caption coverage
        if "image" in self.condition_types:
            assert len(masks) == len(
                ims
            ), "Condition Type Image but could not find masks for all images"  # Validate mask coverage
        print("Found {} images".format(len(ims)))  # Log number of images found
        print("Found {} masks".format(len(masks)))  # Log number of masks found
        print(
            "Found {} captions".format(len(texts))
        )  # Log number of caption sets found
        return ims, texts, masks  # Return lists for downstream indexing

    def get_mask(self, index):
        r"""
        Method to get the mask of WxH
        for given index and convert it into
        Classes x W x H mask image
        :param index:
        :return:
        """
        mask_im = Image.open(self.masks[index])  # Load the mask image file
        mask_im = np.array(
            mask_im
        )  # Convert mask to numpy array for pixel-wise operations
        im_base = np.zeros(
            (self.mask_h, self.mask_w, self.mask_channels)
        )  # Initialize empty one-hot mask volume
        for orig_idx in range(len(self.idx_to_cls_map)):
            im_base[mask_im == (orig_idx + 1), orig_idx] = (
                1  # Turn on voxels where this class label appears
            )
        mask = (
            torch.from_numpy(im_base).permute(2, 0, 1).float()
        )  # Convert to CHW tensor and cast to float
        return mask  # Return tensor suitable for conditioning

    def __len__(self):
        return len(self.images)  # Dataset length equals number of available images

    def __getitem__(self, index):
        ######## Set Conditioning Info ########
        cond_inputs = {}  # Prepare container for conditional signals
        if "text" in self.condition_types:
            cond_inputs["text"] = random.sample(self.texts[index], k=1)[
                0
            ]  # Sample a single caption for classifier-free guidance
        if "image" in self.condition_types:
            mask = self.get_mask(index)  # Create segmentation mask tensor
            cond_inputs["image"] = mask  # Attach mask to conditioning dict
        #######################################

        if self.use_latents:
            latent = self.latent_maps[
                self.images[index]
            ]  # Look up latent tensor by image path key
            if len(self.condition_types) == 0:
                return latent  # Return only latent when no conditioning is requested
            else:
                return latent, cond_inputs  # Return latent with conditioning payload
        else:
            im = Image.open(self.images[index])  # Load raw image file
            im_tensor = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(
                        self.im_size
                    ),  # Resize shortest side to target size
                    torchvision.transforms.CenterCrop(
                        self.im_size
                    ),  # Center crop to square
                    torchvision.transforms.ToTensor(),  # Convert to normalized tensor in [0, 1]
                ]
            )(im)
            im.close()  # Close file descriptor

            # Convert input to -1 to 1 range.
            im_tensor = (2 * im_tensor) - 1  # Rescale tensor from [0, 1] to [-1, 1]
            if len(self.condition_types) == 0:
                return im_tensor  # Return image only when unconditional
            else:
                return (
                    im_tensor,
                    cond_inputs,
                )  # Return image plus conditioning dictionary
