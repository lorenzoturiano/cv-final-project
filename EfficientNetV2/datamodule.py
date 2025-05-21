import os
from typing import Optional, Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# class FullDescriptionDataset(Dataset):
#     def __init__(self, background_dir: str, images_dir: str, masks_dir: str, common_transform=None, image_transform=None, mask_divisor=False):
#         self.background_dir = background_dir
#         self.images_dir = images_dir
#         self.masks_dir = masks_dir
#         self.common_transform = common_transform
#         self.image_transform = image_transform
#         self.backgrounds = sorted(os.listdir(background_dir))
#         self.images = sorted(os.listdir(images_dir))
#         self.masks = sorted(os.listdir(masks_dir))
#         self.mask_divisor = mask_divisor

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         background_path = os.path.join(self.background_dir, self.backgrounds[idx])
#         img_path = os.path.join(self.images_dir, self.images[idx])
#         mask_path = os.path.join(self.masks_dir, self.masks[idx])
#         background = Image.open(background_path).convert("L")  # grayscale PIL image
#         image = Image.open(img_path).convert("L")  # grayscale PIL image
#         mask = Image.open(mask_path).convert("L")  # grayscale PIL image

#         background = np.array(background)[..., None]  # shape (H, W, 1)
#         image = np.array(image)[..., None]  # shape (H, W, 1)
#         mask = np.array(mask)[..., None]    # shape (H, W, 1)
#         if self.common_transform:
#             augmented = self.common_transform(background=background, image=image, mask=mask)
#             background = augmented["background"]
#             image = augmented["image"]
#             mask = augmented["mask"]
#         if self.image_transform:
#             augmented = self.image_transform(background=background, image=image)
#             background = augmented["background"]
#             image = augmented["image"]

#         mask = mask.astype(np.float32)/255.0
#         image = image.astype(np.float32)/255.0
#         background = background.astype(np.float32)/255.0
        
#         image = np.concatenate([image, image, image], axis=-1)  # Convert to 3 channels
#         # permute image and mask to (C, H, W)
#         image = np.transpose(image, (2, 0, 1))
#         mask = np.transpose(mask, (2, 0, 1))
#         background = np.transpose(background, (2, 0, 1))

#         return background, image, mask



import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class FullDescriptionDataset(Dataset):
    def __init__(self, background_dir, images_dir, masks_dir, common_transform=None, image_transform=None, mask_divisor=False):
        self.background_paths = sorted([os.path.join(background_dir, f) for f in os.listdir(background_dir)])
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
        self.mask_paths = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])
        self.common_transform = common_transform
        self.image_transform = image_transform
        self.mask_divisor = mask_divisor

    def __len__(self):
        return len(self.image_paths)

    def _load_grayscale(self, path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape (H, W), dtype=uint8
        return img[..., None]  # shape (H, W, 1)

    def __getitem__(self, idx):
        background = self._load_grayscale(self.background_paths[idx])
        image = self._load_grayscale(self.image_paths[idx])
        mask = self._load_grayscale(self.mask_paths[idx])

        # Normalizza e converte a float32
        image = image.astype(np.float32) / 255.0
        background = background.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0

        # Replica 1 canale → 3 canali (grayscale → RGB fake)
        image = np.repeat(image, 3, axis=-1)

        # Cambia da HWC → CHW per PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1)
        background = torch.from_numpy(background).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)

        return background, image, mask





class FullDescriptionDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        train_background_dir: str,
        train_images_dir: str,
        train_masks_dir: str,
        val_background_dir: str,
        val_images_dir: str,
        val_masks_dir: str,
        transforms: dict,
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (384, 384),
    ):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.train_background_dir = train_background_dir
        self.val_background_dir = val_background_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        self.train_common_transform = transforms["train_common"]
        self.val_common_transform = transforms["val_common"]
        self.train_image_transform = transforms["train_image"]
        self.val_image_transform = transforms["val_image"]

        

    def setup(self, ):  #stage: Optional[str] = None):
        self.train_dataset = FullDescriptionDataset(
            self.train_background_dir,
            self.train_images_dir,
            self.train_masks_dir,
            common_transform=self.train_common_transform,
            image_transform=self.train_image_transform,
        )
        self.val_dataset = FullDescriptionDataset(
            self.val_background_dir,
            self.val_images_dir,
            self.val_masks_dir,
            common_transform=self.val_common_transform,
            image_transform=self.val_image_transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )



if __name__ == "__main__":
    # Define paths
    IMG_DIM = 384

    background_dir = "../image_generation/generated/backgrounds"
    images_dir = "../image_generation/generated/stretched"
    masks_dir = "../image_generation/generated/masks"

    background_dir = "smalldataset/backgrounds"
    images_dir = "smalldataset/stretched"
    masks_dir = "smalldataset/masks"



    # common_transforms = A.Compose([
    #     A.ShiftScaleRotate(p=0.2),
    #     A.RandomCrop(height=IMG_DIM, width=IMG_DIM),
    #     A.HorizontalFlip(p=0.5),
    #     # ToTensorV2()
    # ], additional_targets={'background': 'image'})

    # # Not for mask
    # image_transforms = A.Compose([
    #     A.RandomBrightnessContrast(p=0.2),  # Solo per l'immagine
    #     # ToTensorV2()
    # ], additional_targets={'background': 'image'})

    # create dataset
    dataset = FullDescriptionDataset(
        background_dir=background_dir,
        images_dir=images_dir,
        masks_dir=masks_dir,
        common_transform=None,
        image_transform=None,
        mask_divisor=False
    )
    # show the first image and mask
    for background, image, mask in dataset:
        mask = mask.cpu().detach().numpy()
        background = background.cpu().detach().numpy()
        image = image.cpu().detach().numpy()
        background = np.transpose(background, (1, 2, 0))
        image = np.transpose(image, (1, 2, 0))
        mask = np.transpose(mask, (1, 2, 0))
        Image.fromarray((background * 255).astype(np.uint8).squeeze()).show(title="background")
        Image.fromarray((image * 255).astype(np.uint8).squeeze()).show(title="image")
        Image.fromarray((mask * 255).astype(np.uint8).squeeze()).show(title="mask")
        # Image.fromarray(mask.squeeze()).show(title="mask")
        break






