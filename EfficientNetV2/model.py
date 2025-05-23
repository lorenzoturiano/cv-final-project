import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s  # EfficientNetV2 Small (per base usiamo v2_m se disponibile)
from torchvision.models.efficientnet import EfficientNet_V2_M_Weights
from torchvision.models.efficientnet import EfficientNet_V2_S_Weights
from torchvision import transforms
import torch.nn.functional as F




class EfficientNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Carichiamo EfficientNetV2-M (base) pretrained
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.model = efficientnet_v2_s(weights=weights)
        # self.preprocess = weights.transforms()        # reduces the size to 384x384
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    
    def forward(self, x):

        skips = []
        skips.append(x)  # skip0 (original image)
        x = torch.stack([self.preprocess(img) for img in x])
        skips.append(x)  # skip1 (preprocessed image)
        x = self.model.features[0](x)
        x = self.model.features[1](x)
        skips.append(x)  # skip2
        
        x = self.model.features[2](x)
        skips.append(x)  # skip3
        
        x = self.model.features[3](x)
        skips.append(x)  # skip4
        
        x = self.model.features[4](x)
        skips.append(x)  # skip5
        
        x = self.model.features[5](x)
        # features[6] è l'ultimo blocco con global pooling, non serve
        x = self.model.features[6](x)
        
        return x, skips

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)        
        x = self.conv(x)
        return x

class UNetEfficientNetV2(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.encoder = EfficientNetV2Encoder(pretrained=True)
        # Encoder: EfficientNetV2-S                                                         # dimensions are true only if done with 256x256 input
        self.decoder5 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=256)  # da 8x8 a 16x16
        self.decoder4 = DecoderBlock(in_channels=256, skip_channels=64, out_channels=128)   # 16x16 -> 32x32
        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=48, out_channels=64)    # 32x32 -> 64x64
        self.decoder2 = DecoderBlock(in_channels=64, skip_channels=24, out_channels=32)     # 64x64 -> 128x128
        self.decoder1_seg = DecoderBlock(in_channels=32, skip_channels=3, out_channels=16)     # 128x128 -> 256x256
        self.decoder1_rec = DecoderBlock(in_channels=32, skip_channels=3, out_channels=16)     # 128x128 -> 256x256


        
        self.segmentation_head = nn.Conv2d(16, n_classes, kernel_size=1)

        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(16, 8 , kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )





        
    def forward(self, x):
        x, skips = self.encoder(x)
        # skips: [skip0, skip1, skip2, skip3, skip4]
        x = self.decoder5(x, skips[5])  
        x = self.decoder4(x, skips[4])  
        x = self.decoder3(x, skips[3])  
        x = self.decoder2(x, skips[2])
        x_seg = self.decoder1_seg(x, skips[1])
        x_seg = self.segmentation_head(x_seg)
        x_rec = self.decoder1_rec(x, skips[0])
        x_rec = self.reconstruction_head(x_rec)
        x_rec = skips[0][:,0:1,:,:] - x_rec           # AGGIUNTA DOPO
        x_rec = torch.clamp(x_rec, 0, 1)    # AGGIUNTA DOPO

        return x_seg, x_rec

# Test modello con input batch 1, 1 canale, 256x256
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetEfficientNetV2(n_classes=1).to(device)
    x = torch.rand(1, 1, 384, 384).to(device) 
    x = torch.cat([x, x, x], dim=1)


    binary_mask = torch.randint(0, 2, (1, 1, 384, 384)).float().to(device)  # Binary mask
    # x = preprocess(x)  # Preprocess input
    x_seg, x_rec, loss = model(x, binary_mask, x[:, 0:1, :, :]) 
    print(x_seg.shape) 
