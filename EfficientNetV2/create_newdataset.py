import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

# === CONFIG ===
INPUT_IMAGES_DIR = "../image_generation/generated/stretched"
INPUT_MASKS_DIR = "../image_generation/generated/masks"
INPUT_BACKGROUNDS_DIR = "../image_generation/generated/backgrounds"
OUTPUT_DIR = "smalldataset"
N_AUGS = 50

IMG_DIM = 384

# === CREAZIONE CARTELLE DESTINAZIONE ===
for sub in ['stretched', 'masks', 'backgrounds']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)



common_transforms = A.Compose([
        A.ShiftScaleRotate(p=0.2),
        A.RandomCrop(height=IMG_DIM, width=IMG_DIM),
        A.HorizontalFlip(p=0.5),
        # ToTensorV2()
    ], additional_targets={'background': 'image'})

    # Not for mask
image_transforms = A.Compose([
        A.RandomBrightnessContrast(p=0.2),  # Solo per l'immagine
        # ToTensorV2()
    ], additional_targets={'background': 'image'})


# === FILE ===
images = sorted(os.listdir(INPUT_IMAGES_DIR))
masks = sorted(os.listdir(INPUT_MASKS_DIR))
backgrounds = sorted(os.listdir(INPUT_BACKGROUNDS_DIR))

assert len(images) == len(masks) == len(backgrounds), "Mismatch nei file"

# === LOOP PRINCIPALE ===
for i in tqdm(range(len(images))):
    img_path = os.path.join(INPUT_IMAGES_DIR, images[i])
    mask_path = os.path.join(INPUT_MASKS_DIR, masks[i])
    bg_path = os.path.join(INPUT_BACKGROUNDS_DIR, backgrounds[i])

    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[..., None]
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)[..., None]
    background = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)[..., None]

    for j in range(N_AUGS):
        augmented = common_transforms(image=image, mask=mask, background=background)
        img_aug = augmented["image"]
        mask_aug = augmented["mask"]
        bg_aug = augmented["background"]
        # Applica le trasformazioni specifiche per l'immagine
        augmented = image_transforms(image=img_aug, background=bg_aug)
        img_aug = augmented["image"]
        bg_aug = augmented["background"]

        # Salva con nomi univoci
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'stretched', f"img_{i}_{j}.png"), img_aug)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'masks', f"mask_{i}_{j}.png"), mask_aug)
        cv2.imwrite(os.path.join(OUTPUT_DIR, 'backgrounds', f"bg_{i}_{j}.png"), bg_aug)
