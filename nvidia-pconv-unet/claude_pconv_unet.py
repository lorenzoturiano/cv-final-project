import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d

class PartialConvBlock(nn.Module):
    """
    A block consisting of a Partial Convolution followed by ReLU and BatchNorm
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(PartialConvBlock, self).__init__()
        self.pconv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding, 
                                  return_mask=True, multi_channel=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.relu(x)
        x = self.bn(x)
        return x, mask

class PartialConvUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(PartialConvUNet, self).__init__()
        
        # Encoder path (downsampling)
        # First level: 572x572 -> 570x570 (conv 3x3) -> 568x568 (conv 3x3) -> 284x284 (maxpool 2x2)
        self.enc1_1 = PartialConvBlock(input_channels, 64, kernel_size=3, padding=0)
        self.enc1_2 = PartialConvBlock(64, 64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second level: 284x284 -> 282x282 -> 280x280 -> 140x140
        self.enc2_1 = PartialConvBlock(64, 128, kernel_size=3, padding=0)
        self.enc2_2 = PartialConvBlock(128, 128, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third level: 140x140 -> 138x138 -> 136x136 -> 68x68
        self.enc3_1 = PartialConvBlock(128, 256, kernel_size=3, padding=0)
        self.enc3_2 = PartialConvBlock(256, 256, kernel_size=3, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fourth level: 68x68 -> 66x66 -> 64x64 -> 32x32
        self.enc4_1 = PartialConvBlock(256, 512, kernel_size=3, padding=0)
        self.enc4_2 = PartialConvBlock(512, 512, kernel_size=3, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottom level: 32x32 -> 30x30 -> 28x28
        self.bottom_1 = PartialConvBlock(512, 1024, kernel_size=3, padding=0)
        self.bottom_2 = PartialConvBlock(1024, 1024, kernel_size=3, padding=0)
        
        # Decoder path (upsampling)
        # First upsampling: 28x28 -> 56x56 (upconv)
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # Note: Handling the mask upsampling separately in forward()
        
        # After concatenation, we have 512 (upsampled) + 512 (skip) = 1024 channels
        self.dec1_1 = PartialConvBlock(1024, 512, kernel_size=3, padding=0)
        self.dec1_2 = PartialConvBlock(512, 512, kernel_size=3, padding=0)
        
        # Second upsampling
        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # After concatenation, we have 256 (upsampled) + 256 (skip) = 512 channels
        self.dec2_1 = PartialConvBlock(512, 256, kernel_size=3, padding=0)
        self.dec2_2 = PartialConvBlock(256, 256, kernel_size=3, padding=0)
        
        # Third upsampling
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # After concatenation, we have 128 (upsampled) + 128 (skip) = 256 channels
        self.dec3_1 = PartialConvBlock(256, 128, kernel_size=3, padding=0)
        self.dec3_2 = PartialConvBlock(128, 128, kernel_size=3, padding=0)
        
        # Fourth upsampling
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # After concatenation, we have 64 (upsampled) + 64 (skip) = 128 channels
        self.dec4_1 = PartialConvBlock(128, 64, kernel_size=3, padding=0)
        self.dec4_2 = PartialConvBlock(64, 64, kernel_size=3, padding=0)
        
        # Final 1x1 convolution to generate segmentation map
        self.final = nn.Conv2d(64, output_channels, kernel_size=1)
        
    def forward(self, x, mask):
        # Encoder path
        # Level 1
        e1_1, m1_1 = self.enc1_1(x, mask)
        e1_2, m1_2 = self.enc1_2(e1_1, m1_1)
        pool1 = self.pool1(e1_2)
        pool1_mask = self.pool1(m1_2)
        
        # Level 2
        e2_1, m2_1 = self.enc2_1(pool1, pool1_mask)
        e2_2, m2_2 = self.enc2_2(e2_1, m2_1)
        pool2 = self.pool2(e2_2)
        pool2_mask = self.pool2(m2_2)
        
        # Level 3
        e3_1, m3_1 = self.enc3_1(pool2, pool2_mask)
        e3_2, m3_2 = self.enc3_2(e3_1, m3_1)
        pool3 = self.pool3(e3_2)
        pool3_mask = self.pool3(m3_2)
        
        # Level 4
        e4_1, m4_1 = self.enc4_1(pool3, pool3_mask)
        e4_2, m4_2 = self.enc4_2(e4_1, m4_1)
        pool4 = self.pool4(e4_2)
        pool4_mask = self.pool4(m4_2)
        
        # Bottom
        b1, mb1 = self.bottom_1(pool4, pool4_mask)
        b2, mb2 = self.bottom_2(b1, mb1)
    
    def center_crop(self, tensor, target_size):
        """
        Center crops a tensor to the target size
        Args:
            tensor: input tensor to crop
            target_size: (height, width) tuple of target size
        """
        _, _, h, w = tensor.shape
        th, tw = target_size
        
        # Calculate starting indices for crop
        i = (h - th) // 2
        j = (w - tw) // 2
        
        # Perform the center crop
        return tensor[:, :, i:i+th, j:j+tw]


# Example usage
if __name__ == "__main__":
    # Create a model instance
    model = PartialConvUNet(input_channels=1, output_channels=2)
    
    # Example input (using 572x572 as in the diagram)
    batch_size = 4
    channels = 1
    height = 572
    width = 572
    
    # Create a random input image and mask
    x = torch.randn(batch_size, channels, height, width)
    mask = torch.ones(batch_size, channels, height, width)  # Example: full mask (no holes)
    
    # Add some random holes to the mask (0s represent holes)
    mask[:, :, 200:300, 200:300] = 0  # Create a square hole
    
    # Apply the model
    output = model(x, mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output shape: {output.shape}")