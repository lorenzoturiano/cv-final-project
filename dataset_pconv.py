from pathlib import Path
from typing import Tuple, List
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class StarRemovalGrayDatasetCV(Dataset):
    """Dataset that yields *(masked_image, mask, ground_truth)* tuples
    ready for **partial‑convolution U‑Net training**.

    The directory structure is assumed to be:

    ````text
    root/
        images/  *.png           # uncorrupted grayscale images
        masks/   *.png           # binary masks (white = keep, black = hole)
    ````

    **Returns** each sample with shapes *(1, H, W)*, *(1, H, W)*, *(1, H, W)*
    and value ranges:

    * *masked_image* ∈ **[-1, 1]** (pre‑scaled)
    * *mask*         ∈ {0, 1}
    * *ground_truth* ∈ **[-1, 1]**
    """

    def __init__(self,
                 root: str | Path,
                 crop_size: int = 256,
                 training: bool = True):
        super().__init__()
        root = Path(root)
        self.img_paths: List[Path]  = sorted((root / "images").glob("*.png"))
        self.mask_paths: List[Path] = sorted((root / "masks").glob("*.png"))
        assert len(self.img_paths) == len(self.mask_paths), "#images != #masks"
        self.crop_size = crop_size
        self.training  = training

    # ---------------------------------------------------------------
    def __len__(self):
        return len(self.img_paths)

    # ---------------------------------------------------------------
    def _random_crop(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies the *same* random crop to image and mask."""
        _, H, W = img.shape
        if H == self.crop_size and W == self.crop_size:
            return img, mask
        top  = random.randint(0, H - self.crop_size)
        left = random.randint(0, W - self.crop_size)
        img  = img[:, top:top + self.crop_size, left:left + self.crop_size]
        mask = mask[:, top:top + self.crop_size, left:left + self.crop_size]
        return img, mask

    # ---------------------------------------------------------------
    def __getitem__(self, idx: int):
        # --------- load --------------------------------------------------
        img_p  = self.img_paths[idx]
        mask_p = self.mask_paths[idx]
        img  = Image.open(img_p).convert("L")   # grayscale
        mask = Image.open(mask_p).convert("L")

        # --------- to tensor -------------------------------------------
        img_t  = TF.to_tensor(img)   # (1, H, W) in [0, 1]
        mask_t = TF.to_tensor(mask)  # (1, H, W) in [0, 1]

        # --------- binarise mask (white=keep, black=hole) -------------
        mask_bin = (mask_t > 0.5).float()

        # --------- optional random crop ------------------------------
        if self.training:
            img_t, mask_bin = self._random_crop(img_t, mask_bin)

        # --------- make masked input ---------------------------------
        masked_img = img_t * mask_bin

        # --------- scale images to [-1, 1] ----------------------------
        img_scaled      = img_t * 2.0 - 1.0       # ground truth
        masked_img_sc   = masked_img * 2.0 - 1.0  # input to network

        return masked_img_sc, mask_bin, img_scaled


# =====================================================================
# Quick sanity check ---------------------------------------------------
# python dataset_pconv.py /path/to/data
# =====================================================================
if __name__ == "__main__":
    import sys
    from torch.utils.data import DataLoader

    if len(sys.argv) < 2:
        print("Usage: python dataset_pconv.py <root_dir>")
        sys.exit(0)

    ds = StarRemovalGrayDatasetCV(sys.argv[1])
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    masked, mask, gt = next(iter(loader))
    print("masked:", masked.shape, masked.min().item(), masked.max().item())
    print("mask:  ", mask.shape, mask.unique())
    print("gt:    ", gt.shape, gt.min().item(), gt.max().item())
