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



class BinarySegmentationDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, common_transform=None, image_transform=None, mask_divisor = False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.common_transform = common_transform
        self.image_transform = image_transform
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.mask_divisor = mask_divisor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # shape (H, W)
        image = image[:, :, None]  # shape (H, W, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # shape (H, W)
        mask = mask[:, :, None]  # shape (H, W, 1)

        # Applica le trasformazioni comuni (composte per immagine e maschera)
        if self.common_transform:
            augmented = self.common_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        if self.image_transform:
            # Applica le trasformazioni specifiche solo all'immagine (es. luminosità/contrasto)
            image = self.image_transform(image=image)["image"]
        if self.mask_divisor:
            mask = mask // 255
        mask = torch.from_numpy((mask > 0.5)).float().permute(2, 0, 1)  # Ensure binary
        return image, mask

class BinarySegmentationDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_images_dir: str,
        train_masks_dir: str,
        val_images_dir: str,
        val_masks_dir: str,
        transforms: dict,
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (600, 600),
        mask_divisor: bool = False
    ):
        super().__init__()
        self.train_images_dir = train_images_dir
        self.train_masks_dir = train_masks_dir
        self.val_images_dir = val_images_dir
        self.val_masks_dir = val_masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        self.train_common_transform = transforms["train_common"]
        self.val_common_transform = transforms["val_common"]
        self.train_image_transform = transforms["train_image"]
        self.val_image_transform = transforms["val_image"]
        self.mask_divisor = mask_divisor

        

    def setup(self, ):  #stage: Optional[str] = None):
        self.train_dataset = BinarySegmentationDataset(
            self.train_images_dir,
            self.train_masks_dir,
            common_transform=self.train_common_transform,
            image_transform=self.train_image_transform,
            mask_divisor= self.mask_divisor
        )
        self.val_dataset = BinarySegmentationDataset(
            self.val_images_dir,
            self.val_masks_dir,
            common_transform=self.val_common_transform,
            image_transform=self.val_image_transform,
            mask_divisor= self.mask_divisor
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
