from model import UNetEfficientNetV2
from datamodule import FullDescriptionDatamodule

import albumentations as A
import torch
import numpy as np
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import cv2


def loss_fn_rec(x_rec, mask, background,
                    lambda_valid=1.0, lambda_hole=6.0, lambda_tv=0.2):
    """
    x_rec: modello (B, C, H, W)
    mask: binary mask (1=buco, 0=valido), (B, 1, H, W)
    background: ground truth (B, C, H, W)
    """

    hole = mask
    gamma = 0.5
    l1_valid = F.l1_loss(x_rec, background)
    l2_valid = F.mse_loss(x_rec, background)
    # l2_hole = F.mse_loss(x_rec * hole, background * hole)

    # tv_h = torch.mean(torch.abs(x_rec[:, :, 1:, :] - x_rec[:, :, :-1, :]))
    # tv_w = torch.mean(torch.abs(x_rec[:, :, :, 1:] - x_rec[:, :, :, :-1]))
    # tv = tv_h + tv_w

    # oppure
    # diff = (x_rec - background) ** 2
    diff = torch.abs(x_rec - background)
    masked_diff = diff * hole
    l2_hole = masked_diff.sum() / hole.sum()

    loss = lambda_valid * (gamma * l1_valid + (1 - gamma) * l2_valid) + lambda_hole * l2_hole       #+ lambda_tv * tv
    return loss



def soft_erosion_mask(mask, kernel_size=9, sigma=0):
    """
    Applica una soft erosion alla maschera, preservando le aree già forti.
    mask: array con valori float32 tra 0 e 1, shape (H, W) o (H, W, 1)
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask.squeeze(-1)

    blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    combined = np.maximum(mask, blurred)
    
    return combined[..., np.newaxis]  # Ritorna con shape (H, W, 1)


def test_on_val(model, val_loader, device):
    MAX = 50
    model.eval()
    total_loss_rec = 0
    total_loss_seg = 0
    for i, (background, image, mask) in enumerate(val_loader):
        image = image.to(device)
        mask = mask.to(device)
        background = background.to(device)
        x_seg, x_rec = model(image)
        loss_rec = loss_fn_rec(x_rec, mask, background, lambda_hole=2.0)
        loss_seg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(30).to(device))(x_seg, mask)
        total_loss_rec += loss_rec.item()
        total_loss_seg += loss_seg.item()
        if i >= MAX:
            break
    avg_loss_rec = total_loss_rec / min(MAX, len(val_loader))
    avg_loss_seg = total_loss_seg / min(MAX, len(val_loader))
    print(f"Reconstruction Loss: {avg_loss_rec}, Segmentation Loss: {avg_loss_seg}")
    return



if __name__ == "__main__":
    # Define paths
    IMG_DIM = 384

    # background_dir = "../image_generation/generated/backgrounds"
    # images_dir = "../image_generation/generated/stretched"
    # masks_dir = "../image_generation/generated/masks"


    # background_dir = "smallRIC/backgrounds"
    # images_dir = "smallRIC/images"
    # masks_dir = "smallRIC/masks"

    background_dir = "smallreal2/backgrounds"
    images_dir = "smallreal2/stretched"
    masks_dir = "smallreal2/masks"

    val_background_dir = "val_smallRIC/backgrounds"
    val_images_dir = "val_smallRIC/images"
    val_masks_dir = "val_smallRIC/masks"

    # background_dir = "smalldataset/backgrounds"
    # images_dir = "smalldataset/stretched"
    # masks_dir = "smalldataset/masks"


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
    dm = FullDescriptionDatamodule(
        train_background_dir=background_dir,
        train_images_dir=images_dir,
        train_masks_dir=masks_dir,
        val_background_dir=val_background_dir,
        val_images_dir=val_images_dir,
        val_masks_dir=val_masks_dir,
        # val_background_dir=background_dir,
        # val_images_dir=images_dir,
        # val_masks_dir=masks_dir,


        transforms= {"train_common": None,
                    "train_image": None,
                    "val_common": None,
                    "val_image": None},
        img_size=(384, 384),
        batch_size=4,           # MODIFICATO ERA 8
            num_workers=0,      # SE aumento il numero di workers, va molto più lento
    )
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()


    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEfficientNetV2().to(device)
    # Load pretrained weights
    #last best 136 (old version)
    # 9309 quello nuovo
    # MODELLO OTTIMO: 5476, pre reali
    model.load_state_dict(torch.load("checkpoints/model_481197.pth"))
    # freeze the encoder
    # for param in model.encoder.parameters():
    #     param.requires_grad = False
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    # JUST NOW

    def reset_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            m.reset_parameters()

    # model.reconstruction_head.apply(reset_weights)
    # model.decoder1_rec.apply(reset_weights)

    alpha = 0.1
    loss_fn_seg = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20).to(device))
    name = np.random.randint(0, 1000000)
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = 0
        for i, (background, image, mask) in enumerate(train_loader):
            image = image.to(device)
            mask = mask.to(device)
            background = background.to(device)
            x_seg, x_rec = model(image)
            loss = alpha * loss_fn_seg(x_seg, mask) + (1 - alpha) * loss_fn_rec(x_rec, mask, background, lambda_hole=3.0)
            # if i == 0:
            #     print("seg: ", loss_fn_seg(x_seg, mask), "rec: ", loss_fn_rec(x_rec, mask, background, lambda_hole=8.0))
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader)}")
        test_on_val(model, val_loader, device)
        #Save model
        torch.save(model.state_dict(), f"checkpoints/model_{name}.pth")



    # Test model
    for i, (background, image, mask) in enumerate(val_loader):   
        image = image.to(device)
        mask = mask.to(device)
        background = background.to(device)
        x_seg, x_rec = model(image)
        # plot results
        x_seg = x_seg.cpu().detach().numpy()
        x_rec = x_rec.cpu().detach().numpy()
        mask = mask.cpu().detach().numpy()
        background = background.cpu().detach().numpy()
        image = image.cpu().detach().numpy()

        x_seg = np.transpose(x_seg, (0, 2, 3, 1))
        x_rec = np.transpose(x_rec, (0, 2, 3, 1))
        mask = np.transpose(mask, (0, 2, 3, 1))
        background = np.transpose(background, (0, 2, 3, 1))
        image = np.transpose(image, (0, 2, 3, 1))
        # apply sigmoid to x_seg
        x_seg = 1 / (1 + np.exp(-x_seg))
        x_rec = (x_rec * 255).astype(np.uint8)
        mask = (mask * 255).astype(np.uint8)
        background = (background * 255).astype(np.uint8)
        image = (image * 255).astype(np.uint8)
        x_rec = x_rec[..., 0]
        for j in range(x_seg.shape[0]):
            # x_seg[j] = cv2.GaussianBlur(x_seg[j], (9, 9), 0)[..., np.newaxis]
            x_seg[j] = soft_erosion_mask(x_seg[j], kernel_size=7, sigma=1)
            x_rec[j] = cv2.GaussianBlur(x_rec[j], (3, 3), 1)
            noise = np.random.normal(0, 1, size=x_rec[j].shape)  # rumore gaussiano     prima era 5, ora lo metto a 1
            x_rec[j] = np.clip(x_rec[j] + noise, 0, 255)

            ############### DA RIMETTERE A PROST
        x_rec = x_rec[..., np.newaxis]

        final = (x_rec * x_seg + image * (1 - x_seg)).astype(np.uint8)
        x_seg = (x_seg * 255).astype(np.uint8)

        for j in range(x_seg.shape[0]):
            Image.fromarray(x_seg[j].squeeze()).save(f"results/segmentation_{i}_{j}.png")
            Image.fromarray(x_rec[j].squeeze()).save(f"results/reconstruction_{i}_{j}.png")
            Image.fromarray(mask[j].squeeze()).save(f"results/mask_{i}_{j}.png")
            Image.fromarray(background[j].squeeze()).save(f"results/background_{i}_{j}.png")
            Image.fromarray(image[j].squeeze()).save(f"results/image_{i}_{j}.png")
            Image.fromarray(final[j].squeeze()).save(f"results/final_{i}_{j}.png")

            if j >= 3:
                break
        break
        



