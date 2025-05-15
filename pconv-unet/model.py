import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class StarRemovalGrayDataset(Dataset):
    """
    Returns:
        input_tensor  - shape (2, H, W): [masked_image, binary_mask]
        target_tensor - shape (1, H, W): ground-truth star-free image
    """
    def __init__(self, masked_dir, mask_dir, gt_dir,
                 img_size=(256, 256),
                 threshold=0.5):
        self.masked_paths = sorted(
            [os.path.join(masked_dir, f) for f in os.listdir(masked_dir)]
        )
        self.mask_paths = sorted(
            [os.path.join(mask_dir, f) for f in os.listdir(mask_dir)]
        )
        self.gt_paths = sorted(
            [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]
        )

        # Same transform for all images (grayscale → tensor → resize/normalize)
        self.to_tensor = T.Compose([
            T.Resize(img_size, interpolation=Image.BILINEAR),
            T.ToTensor(),                          # (1, H, W) in [0, 1]
        ])
        self.threshold = threshold                # for binarising masks

    def __len__(self):
        return len(self.masked_paths)

    def __getitem__(self, idx):
        # 1) Masked (grayscale) image with star pixels already zeroed
        masked_img = Image.open(self.masked_paths[idx]).convert("L")
        masked_tensor = self.to_tensor(masked_img)             # (1, H, W)

        # 2) Binary mask (1 = hole, 0 = valid)
        mask_img = Image.open(self.mask_paths[idx]).convert("L")
        mask_tensor = self.to_tensor(mask_img)
        mask_tensor = (mask_tensor > self.threshold).float()   # binarise

        # 3) Ground-truth star-free image
        gt_img = Image.open(self.gt_paths[idx]).convert("L")
        gt_tensor = self.to_tensor(gt_img)                     # (1, H, W)

        # Stack masked image and mask into 2-channel input
        input_tensor = torch.cat([masked_tensor, mask_tensor], dim=0)

        return input_tensor, gt_tensor

# ds = StarRemovalGrayDataset(
#                             "masked",
#                             "masks",
#                             "gt",
#                             img_size=(256,256)
#                             )
# x, y = ds[0]
# print(x.shape)  # torch.Size([2, 256, 256])
# print(y.shape)  # torch.Size([1, 256, 256])

img = Image.open("image_generation/generated/backgrounds/i1.png")
print(img)
