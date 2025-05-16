import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Module):
    """
    Partial Convolutional Layer as described in the paper:
    'Image Inpainting for Irregular Holes Using Partial Convolutions'
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(PartialConv2d, self).__init__()
        
        # Regular convolution for the image data
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
        # Convolution for the mask with same parameters but different weights
        self.mask_conv = nn.Conv2d(
            1, 1, kernel_size,
            stride=stride, padding=padding, dilation=dilation, bias=False
        )
        
        # Initialize mask_conv with ones
        with torch.no_grad():
            self.mask_conv.weight.fill_(1.0)
            
        # Freeze the mask weights
        self.mask_conv.weight.requires_grad = False
        
        # Bias rescaling term
        self.bias = bias
        
    def forward(self, x, mask):
        """Forward pass
        
        Args:
            x: input tensor of shape [B, C, H, W]
            mask: binary mask tensor of shape [B, 1, H, W] where 1 indicates valid pixels
            
        Returns:
            output: tensor after partial convolution
            updated_mask: updated mask for the next layer
        """
        # Make sure the mask has the same number of channels as input x
        if mask.shape[1] != x.shape[1]:
            mask = mask.repeat(1, x.shape[1], 1, 1)
        
        # Apply convolution to input and mask
        output = self.conv(x * mask)
        mask_output = self.mask_conv(mask[:, 0:1, :, :])  # Use only first channel for mask
        
        # Calculate scaling factor based on mask
        # Add small epsilon to avoid division by zero
        mask_ratio = mask_output / (torch.sum(self.mask_conv.weight) + 1e-8)
        mask_ratio = torch.clamp(mask_ratio, 0, 1)
        
        # Scale the output by the mask ratio where mask != 0
        output = output * mask_ratio
        
        # Create updated mask for next layer (values > 0 become 1)
        updated_mask = torch.clamp(mask_ratio, 0, 1)
        
        return output, updated_mask


class DownSampleBlock(nn.Module):
    """Downsampling block with partial convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, bn=True, activation=True):
        super(DownSampleBlock, self).__init__()
        
        self.pconv = PartialConv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
            
        self.activation = None
        if activation:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        
        if self.bn:
            x = self.bn(x)
            
        if self.activation:
            x = self.activation(x)
            
        return x, mask


class UpSampleBlock(nn.Module):
    """Upsampling block with partial convolutions and skip connections"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bn=True, activation=True):
        super(UpSampleBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pconv = PartialConv2d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding
        )
        
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)
            
        self.activation = None
        if activation:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x, mask):
        # Upsample both input and mask
        x = self.upsample(x)
        mask = self.upsample(mask)
        
        # Apply partial convolution
        x, mask = self.pconv(x, mask)
        
        if self.bn:
            x = self.bn(x)
            
        if self.activation:
            x = self.activation(x)
            
        return x, mask


class PartialConvUNet(nn.Module):
    """UNet-like architecture using partial convolutions for image inpainting"""
    def __init__(self):
        super(PartialConvUNet, self).__init__()
        
        # Encoder
        self.enc1 = DownSampleBlock(1, 32, stride=1)  # Output: [32, 256, 256]
        self.enc2 = DownSampleBlock(32, 64)           # Output: [64, 128, 128]
        self.enc3 = DownSampleBlock(64, 128)          # Output: [128, 64, 64]
        self.enc4 = DownSampleBlock(128, 256)         # Output: [256, 32, 32]
        self.enc5 = DownSampleBlock(256, 512)         # Output: [512, 16, 16]
        
        # Decoder with skip connections
        self.dec5 = UpSampleBlock(512, 256)           # Output: [256, 32, 32]
        self.dec4 = UpSampleBlock(512, 128)           # Output: [128, 64, 64]
        self.dec3 = UpSampleBlock(256, 64)            # Output: [64, 128, 128]
        self.dec2 = UpSampleBlock(128, 32)            # Output: [32, 256, 256]
        
        # Final layer (no batch norm and sigmoid activation)
        self.final = PartialConv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, x, mask):
        """
        Forward pass
        
        Args:
            x: Corrupted image tensor of shape [B, 1, H, W]
            mask: Binary mask tensor of shape [B, 1, H, W] (1 for valid pixels, 0 for holes)
            
        Returns:
            out: Inpainted image
        """
        # Make sure mask has proper dimensions
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # Encoder
        enc1, mask1 = self.enc1(x, mask)
        enc2, mask2 = self.enc2(enc1, mask1)
        enc3, mask3 = self.enc3(enc2, mask2)
        enc4, mask4 = self.enc4(enc3, mask3)
        enc5, mask5 = self.enc5(enc4, mask4)
        
        # Decoder
        dec5, dec_mask5 = self.dec5(enc5, mask5)
        
        # Add skip connections - concatenating feature maps properly
        dec4_input = torch.cat([dec5, enc4], dim=1)           # Channels: 256 + 256 = 512
        dec4_mask_input = torch.cat([dec_mask5, mask4], dim=1) # Ensure this is also concatenated properly
        dec4, dec_mask4 = self.dec4(dec4_input, dec4_mask_input[:, :1, :, :])  # Use only first channel for mask
        
        dec3_input = torch.cat([dec4, enc3], dim=1)           # Channels: 128 + 128 = 256
        dec3_mask_input = torch.cat([dec_mask4, mask3], dim=1)
        dec3, dec_mask3 = self.dec3(dec3_input, dec3_mask_input[:, :1, :, :])
        
        dec2_input = torch.cat([dec3, enc2], dim=1)           # Channels: 64 + 64 = 128
        dec2_mask_input = torch.cat([dec_mask3, mask2], dim=1)
        dec2, dec_mask2 = self.dec2(dec2_input, dec2_mask_input[:, :1, :, :])
        
        # Final layer
        final_input = torch.cat([dec2, enc1], dim=1)           # Channels: 32 + 32 = 64
        final_mask_input = torch.cat([dec_mask2, mask1], dim=1)
        final, _ = self.final(final_input, final_mask_input[:, :1, :, :])
        
        # Apply sigmoid to get values in range [0, 1]
        out = torch.sigmoid(final)
        
        return out