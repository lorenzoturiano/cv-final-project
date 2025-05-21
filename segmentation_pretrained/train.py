"""train.py used to train the model with pretrained weights."""

import os
import cv2
import torch
import pytorch_lightning as pl
from datamodule import BinarySegmentationDataModule
from model import EfficientNetV2Encoder
import albumentations as A
from albumentations.pytorch import ToTensorV2



if __name__ == "__main__":
    # Define paths
    IMG_DIM = 480

    train_images_dir = "../output_blob/generated/stretched"
    train_masks_dir = "../output_blob/generated/masks_man"
    val_images_dir = "../output_blob/generated/stretched"
    val_masks_dir = "../output_blob/generated/masks_man"

    common_transforms = A.Compose([
        A.ShiftScaleRotate(p=0.2),
        A.RandomCrop(height=IMG_DIM, width=IMG_DIM),
        A.HorizontalFlip(p=0.5),
        # ToTensorV2()

    ])

    # Not for mask
    image_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),  # Solo per l'immagine
        ToTensorV2()
    ])

    dm = BinarySegmentationDataModule(
        train_images_dir= train_images_dir,
        train_masks_dir= train_masks_dir,
        val_images_dir= val_images_dir,
        val_masks_dir= val_masks_dir,

        transforms= {"train_common": common_transforms,
                    "train_image": image_transforms,
                    "val_common": common_transforms,
                    "val_image": image_transforms},
        img_size=(IMG_DIM, IMG_DIM),
        batch_size=1,
        num_workers=0,      # SE aumento il numero di workers, va molto più lento
        mask_divisor=True,
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # Initialize model
    model = EfficientNetV2Encoder(pretrained=True)

    # Initialize trainer
    trainer = pl.Trainer(max_epochs=1, gpus=1 if torch.cuda.is_available() else 0)

    # Train the model
    trainer.fit(model, data_module)