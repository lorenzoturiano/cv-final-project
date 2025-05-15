import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PartialConv2d(nn.Module):
    """Partial‑convolution layer (Liu *et al.*, ECCV 2018).

    Works exactly like :class:`torch.nn.Conv2d` but requires (or creates) a **binary mask**
    indicating valid pixels. The mask is updated and the output is re‑normalised
    automatically. Forward:

    ```python
    feat, mask = pconv(x, mask)  # mask can be None → assume all valid
    ```
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int] = 1,
                 padding: int | Tuple[int, int] = 0,
                 dilation: int | Tuple[int, int] = 1,
                 bias: bool = True,
                 multi_channel: bool = False,
                 eps: float = 1e-8):
        super().__init__()
        self.multi_channel = multi_channel
        self.eps = eps

        # Learnable content convolution
        self.input_conv = nn.Conv2d(in_channels, out_channels,
                                    kernel_size, stride, padding, dilation,
                                    bias=bias)

        # Non‑learnable all‑ones kernel for mask propagation
        kh, kw = (kernel_size if isinstance(kernel_size, Tuple) else (kernel_size, kernel_size))
        ones_shape = (in_channels if multi_channel else 1, 1, kh, kw)
        self.register_buffer("weight_maskUpdater", torch.ones(ones_shape))
        self.slide_winsize = kh * kw

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # Create default mask if missing
        if mask is None:
            if self.multi_channel:
                mask = torch.ones_like(x)
            else:
                mask = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)

        # (1) Mask input, (2) convolve
        x_masked = x * mask
        y = self.input_conv(x_masked)

        # (3) Update mask
        with torch.no_grad():
            mask_sum = F.conv2d(mask, self.weight_maskUpdater,
                                stride=self.input_conv.stride,
                                padding=self.input_conv.padding,
                                dilation=self.input_conv.dilation)

        # (4) Renormalise
        mask_ratio = self.slide_winsize / (mask_sum + self.eps)
        mask_ratio = mask_ratio * (mask_sum > 0)
        y = y * mask_ratio

        new_mask = (mask_sum > 0).float()
        return y, new_mask


# ======================================================================
# Building blocks
# ======================================================================
class _PConvBlock(nn.Module):
    """Encoder: *PartialConv → BN → ReLU*. Down‑sampling via stride=2."""
    def __init__(self, in_c: int, out_c: int, stride: int = 2, bn: bool = True):
        super().__init__()
        self.pconv = PartialConv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_c) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, m):
        x, m = self.pconv(x, m)
        x = self.bn(x)
        x = self.act(x)
        return x, m


class _PConvUpBlock(nn.Module):
    """Decoder: *upsample → concat(skip) → PartialConv → BN → ReLU*.

    Masks remain single‑channel — merged with logical OR (``torch.max``).
    """
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.pconv = PartialConv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, m, skip_x, skip_m):
        x = self.upsample(x)
        m = self.upsample(m)
        x = torch.cat([x, skip_x], dim=1)
        m = torch.max(m, skip_m)
        x, m = self.pconv(x, m)
        x = self.bn(x)
        x = self.act(x)
        return x, m


# ======================================================================
# Full PConv‑U‑Net (configurable channels)
# ======================================================================
class PConvUNet(nn.Module):
    """Eight‑level PConv‑U‑Net.

    * `in_channels`  –  #channels of your input image (1 for grayscale, 3 for RGB)
    * `out_channels` –  #channels of the reconstruction (usually same as above)

    The *final‑concat* trick (concatenate last decoder feature with raw input)
    is preserved. Channel counts adapt automatically.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_c = in_channels
        self.out_c = out_channels

        # ---------------- Encoder ----------------
        self.enc1 = _PConvBlock(in_channels, 64, stride=1, bn=False)
        self.enc2 = _PConvBlock(64, 128)
        self.enc3 = _PConvBlock(128, 256)
        self.enc4 = _PConvBlock(256, 512)
        self.enc5 = _PConvBlock(512, 512)
        self.enc6 = _PConvBlock(512, 512)
        self.enc7 = _PConvBlock(512, 512)
        self.enc8 = _PConvBlock(512, 512)

        # ---------------- Decoder ----------------
        self.dec8 = _PConvUpBlock(512 + 512, 512)
        self.dec7 = _PConvUpBlock(512 + 512, 512)
        self.dec6 = _PConvUpBlock(512 + 512, 512)
        self.dec5 = _PConvUpBlock(512 + 512, 512)
        self.dec4 = _PConvUpBlock(512 + 256, 256)
        self.dec3 = _PConvUpBlock(256 + 128, 128)
        self.dec2 = _PConvUpBlock(128 + 64, 64)

        # Final partial conv expects 64 + in_c (because of concat)
        self.final = PartialConv2d(64 + in_channels, out_channels, kernel_size=3, padding=1)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if mask is None:
            mask = torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device, dtype=x.dtype)

        # ---------------- Encoder ----------------
        e1, m1 = self.enc1(x, mask)
        e2, m2 = self.enc2(e1, m1)
        e3, m3 = self.enc3(e2, m2)
        e4, m4 = self.enc4(e3, m3)
        e5, m5 = self.enc5(e4, m4)
        e6, m6 = self.enc6(e5, m5)
        e7, m7 = self.enc7(e6, m6)
        e8, m8 = self.enc8(e7, m7)

        # ---------------- Decoder ----------------
        d8, md8 = self.dec8(e8, m8, e7, m7)
        d7, md7 = self.dec7(d8, md8, e6, m6)
        d6, md6 = self.dec6(d7, md7, e5, m5)
        d5, md5 = self.dec5(d6, md6, e4, m4)
        d4, md4 = self.dec4(d5, md5, e3, m3)
        d3, md3 = self.dec3(d4, md4, e2, m2)
        d2, md2 = self.dec2(d3, md3, e1, m1)

        # ---------------- Final concat ----------------
        d2_cat   = torch.cat([d2, x], dim=1)  # (N, 64 + in_c, H, W)
        mask_cat = torch.max(md2, mask)
        out, _   = self.final(d2_cat, mask_cat)
        return torch.tanh(out)


# ======================================================================
if __name__ == "__main__":
    # ---------- quick sanity check ------------------------------------
    model = PConvUNet(in_channels=1, out_channels=1)  # ← grayscale
    x = torch.randn(4, 1, 256, 256)
    mask = (torch.rand(4, 1, 256, 256) > 0.5).float()
    y = model(x, mask)
    print("Output shape:", y.shape)  # → torch.Size([4, 1, 256, 256])
