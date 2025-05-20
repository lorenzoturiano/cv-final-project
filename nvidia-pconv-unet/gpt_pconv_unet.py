import torch
import torch.nn as nn
import torch.nn.functional as F
from partialconv2d import PartialConv2d


class PartialConvBlock(nn.Module):
    """
    A partial-conv → ReLU → BatchNorm block
    (single-channel mask: multi_channel=False)
    """
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.pconv = PartialConv2d(in_c, out_c, k, s, p,
                                   return_mask=True, multi_channel=False)
        self.relu = nn.ReLU(inplace=True)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, x, mask):
        x, mask = self.pconv(x, mask)
        x = self.relu(x)
        x = self.bn(x)
        return x, mask


class PartialConvUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super().__init__()

        # ───── Encoder ─────
        self.enc1_1 = PartialConvBlock(input_channels, 64)
        self.enc1_2 = PartialConvBlock(64, 64)
        self.pool1  = nn.MaxPool2d(2)

        self.enc2_1 = PartialConvBlock(64, 128)
        self.enc2_2 = PartialConvBlock(128, 128)
        self.pool2  = nn.MaxPool2d(2)

        self.enc3_1 = PartialConvBlock(128, 256)
        self.enc3_2 = PartialConvBlock(256, 256)
        self.pool3  = nn.MaxPool2d(2)

        self.enc4_1 = PartialConvBlock(256, 512)
        self.enc4_2 = PartialConvBlock(512, 512)
        self.pool4  = nn.MaxPool2d(2)

        # ───── Bottleneck ─────
        self.bottom_1 = PartialConvBlock(512, 1024)
        self.bottom_2 = PartialConvBlock(1024, 1024)

        # ───── Decoder ─────
        self.upconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec1_1  = PartialConvBlock(512 + 512, 512)  #  skip + up
        self.dec1_2  = PartialConvBlock(512, 512)

        self.upconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec2_1  = PartialConvBlock(256 + 256, 256)
        self.dec2_2  = PartialConvBlock(256, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3_1  = PartialConvBlock(128 + 128, 128)
        self.dec3_2  = PartialConvBlock(128, 128)

        self.upconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec4_1  = PartialConvBlock(64 + 64, 64)
        self.dec4_2  = PartialConvBlock(64, 64)

        self.final = nn.Conv2d(64, output_channels, 1)

    # ──────────────────────────────────────────────────────────────────
    def forward(self, x, mask):
        # ---------- encoder ----------
        e1_1, m1 = self.enc1_1(x, mask)
        e1_2, m1 = self.enc1_2(e1_1, m1)
        p1, pm1  = self.pool1(e1_2), self.pool1(m1)

        e2_1, m2 = self.enc2_1(p1, pm1)
        e2_2, m2 = self.enc2_2(e2_1, m2)
        p2, pm2  = self.pool2(e2_2), self.pool2(m2)

        e3_1, m3 = self.enc3_1(p2, pm2)
        e3_2, m3 = self.enc3_2(e3_1, m3)
        p3, pm3  = self.pool3(e3_2), self.pool3(m3)

        e4_1, m4 = self.enc4_1(p3, pm3)
        e4_2, m4 = self.enc4_2(e4_1, m4)
        p4, pm4  = self.pool4(e4_2), self.pool4(m4)

        # ---------- bottleneck ----------
        b1, mb = self.bottom_1(p4, pm4)
        b2, mb = self.bottom_2(b1, mb)

        # ---------- decoder (helper) ----------
        def merge_masks(up, skip):
            # both are single-channel tensors in [0,1]
            return torch.logical_or(up.bool(), skip.bool()).float()

        # up-block 1
        u1      = self.upconv1(b2)
        um1     = F.interpolate(mb, scale_factor=2, mode='nearest')
        ce4_2   = self.center_crop(e4_2, u1.shape[2:])
        cm4_2   = self.center_crop(m4,   u1.shape[2:])
        m1_cat  = merge_masks(um1, cm4_2)
        d1_1, m1d = self.dec1_1(torch.cat([u1, ce4_2], 1), m1_cat)
        d1_2, m1d = self.dec1_2(d1_1, m1d)

        # up-block 2
        u2      = self.upconv2(d1_2)
        um2     = F.interpolate(m1d, scale_factor=2, mode='nearest')
        ce3_2   = self.center_crop(e3_2, u2.shape[2:])
        cm3_2   = self.center_crop(m3,   u2.shape[2:])
        m2_cat  = merge_masks(um2, cm3_2)
        d2_1, m2d = self.dec2_1(torch.cat([u2, ce3_2], 1), m2_cat)
        d2_2, m2d = self.dec2_2(d2_1, m2d)

        # up-block 3
        u3      = self.upconv3(d2_2)
        um3     = F.interpolate(m2d, scale_factor=2, mode='nearest')
        ce2_2   = self.center_crop(e2_2, u3.shape[2:])
        cm2_2   = self.center_crop(m2,   u3.shape[2:])
        m3_cat  = merge_masks(um3, cm2_2)
        d3_1, m3d = self.dec3_1(torch.cat([u3, ce2_2], 1), m3_cat)
        d3_2, m3d = self.dec3_2(d3_1, m3d)

        # up-block 4
        u4      = self.upconv4(d3_2)
        um4     = F.interpolate(m3d, scale_factor=2, mode='nearest')
        ce1_2   = self.center_crop(e1_2, u4.shape[2:])
        cm1_2   = self.center_crop(m1,   u4.shape[2:])
        m4_cat  = merge_masks(um4, cm1_2)
        d4_1, m4d = self.dec4_1(torch.cat([u4, ce1_2], 1), m4_cat)
        d4_2, _   = self.dec4_2(d4_1, m4d)

        return self.final(d4_2)

    # ---------- util ----------
    @staticmethod
    def center_crop(t, size_hw):
        _, _, h, w = t.size()
        th, tw = size_hw
        i = (h - th) // 2
        j = (w - tw) // 2
        return t[:, :, i:i+th, j:j+tw]


# quick smoke-test -------------------------------------------------------------
if __name__ == "__main__":
    model = PartialConvUNet()

    x     = torch.randn(1, 1, 256, 256)
    mask  = torch.ones(1, 1, 256, 256)
    mask[:, :, 120:150, 120:150] = 0

    y = model(x, mask)

    print("input:",  x.shape)
    print("mask: ", mask.shape)
    print("output:", y.shape)
