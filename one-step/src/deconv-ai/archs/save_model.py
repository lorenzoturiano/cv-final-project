import torch
from unet import UNetUnconditioned

model = UNetUnconditioned(layer_sizes=[16, 32, 64, 64, 128, 256], image_channels=1)

# Save entire model (architecture + weights)
torch.save(model, "unet.pth")