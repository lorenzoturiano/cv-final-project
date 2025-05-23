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
    diff = (x_rec - background) ** 2
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
    avg_loss_rec = total_loss_rec / max(MAX, len(val_loader))
    avg_loss_seg = total_loss_seg / max(MAX, len(val_loader))
    print(f"Reconstruction Loss: {avg_loss_rec}, Segmentation Loss: {avg_loss_seg}")
    return

def _load_grayscale( path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # shape (H, W), dtype=uint8
        # resize to 384x384
        # img = cv2.resize(img, (384 * 4, 384 * 4), interpolation=cv2.INTER_LINEAR)
        # Crop the center 384x384 region
        h, w = img.shape
        ch, cw = 16 * 60, 16* 60
        # Crop a random 384x384 region
        start_h = np.random.randint(0, h - ch + 1)
        start_w = np.random.randint(0, w - cw + 1)
        img = img[start_h:start_h+ch, start_w:start_w+cw]
        return img[..., None]  # shape (H, W, 1)

if __name__ == "__main__":
    # load image
    img_path = r"D:\Documents\AI\II semestre\Computer Vision\v4 project\cv-final-project\output_blob\p47.png"
    image = _load_grayscale(img_path)


    image = image.astype(np.float32) / 255.0


    # Replica 1 canale → 3 canali (grayscale → RGB fake)
    image = np.repeat(image, 3, axis=-1)

    # Cambia da HWC → CHW per PyTorch
    image = torch.from_numpy(image).permute(2, 0, 1)










    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEfficientNetV2().to(device)
    # Load pretrained weights
    #last best 136 (old version)
    # 9309 quello nuovo
    # MODELLO OTTIMO: 5476, pre reali
    # ultimo usato: 418623
    # ultimo usato: 481197
    model.load_state_dict(torch.load("checkpoints/model_834639.pth"))
    model.eval()
    image = image.to(device)
    image = image.unsqueeze(0)  # Aggiungi batch dimension
    x_seg, x_rec = model(image)

    # plot results
    x_seg = x_seg.cpu().detach().numpy()
    x_rec = x_rec.cpu().detach().numpy()
    image = image.cpu().detach().numpy()

    x_seg = np.transpose(x_seg, (0, 2, 3, 1))
    x_rec = np.transpose(x_rec, (0, 2, 3, 1))
    image = np.transpose(image, (0, 2, 3, 1))
    # apply sigmoid to x_seg
    x_seg = 1 / (1 + np.exp(-x_seg))
    x_rec = (x_rec * 255).astype(np.uint8)
    image = (image * 255).astype(np.uint8)
    x_rec = x_rec[..., 0]
    for j in range(x_seg.shape[0]):
        # x_seg[j] = cv2.GaussianBlur(x_seg[j], (9, 9), 0)[..., np.newaxis]
        x_seg[j] = soft_erosion_mask(x_seg[j], kernel_size=7, sigma=1)
        x_rec[j] = cv2.GaussianBlur(x_rec[j], (3, 3), 1)
        noise = np.random.normal(0, 1, size=x_rec[j].shape)  # rumore gaussiano     prima era 5, ora lo metto a 1
        x_rec[j] = np.clip(x_rec[j] + noise, 0, 255)
    x_rec = x_rec[..., np.newaxis]

    final = (x_rec * x_seg + image * (1 - x_seg)).astype(np.uint8)
    x_seg = (x_seg * 255).astype(np.uint8)
    name = 0

    for j in range(x_seg.shape[0]):
        Image.fromarray(x_seg[j].squeeze()).save(f"results_inf/segmentation_{name}.png")
        Image.fromarray(x_rec[j].squeeze()).save(f"results_inf/reconstruction_{name}.png")
        Image.fromarray(image[j].squeeze()).save(f"results_inf/image_{name}.png")
        Image.fromarray(final[j].squeeze()).save(f"results_inf/final_{name}.png")

    



