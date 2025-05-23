import numpy as np
import torch


class PiecewiseLinear(torch.nn.Module):
    r"""
    Continuous, monotone, learnable piece-wise linear map **and** its inverse.

    Parameters
    ----------
    num_segments : int
        Number of linear pieces (uniformly divides the input range).
    input_range  : (float, float), default (0, 1)
        Domain [x_min, x_max] accepted in *forward* mode.
    output_range : (float, float), default (0, 1)
        Codomain [y_min, y_max] produced in *forward* mode.

    Usage
    -----
    >>> f = PiecewiseLinear(8)
    >>> x = torch.rand(5)
    >>> y = f(x)                     # forward
    >>> x_rec = f(y, inverse=True)   # inverse
    >>> torch.allclose(x, x_rec)     # ≈ 1 e-6 after init
    True
    """

    def __init__(
        self,
        num_segments: int,
        input_range: tuple[float, float] = (0.0, 1.0),
        output_range: tuple[float, float] = (0.0, 1.0),
    ):
        super().__init__()
        if num_segments < 1:
            raise ValueError("num_segments ≥ 1 required")

        x_min, x_max = input_range
        y_min, y_max = output_range
        if not x_max > x_min or not y_max > y_min:
            raise ValueError("Ranges must satisfy max > min")

        self.num_segments = num_segments

        # --- constant break-points -------------------------------------------------
        bp = torch.linspace(x_min, x_max, num_segments + 1)[:-1]  # (n,)
        self.register_buffer("breakpoints", bp, persistent=False)

        # --- learnable positive slope-increments -----------------------------------
        self.raw_delta = torch.nn.Parameter(torch.full((num_segments,), -4.0))

        # --- buffers kept for ONNX graph -------------------------------------------
        self.register_buffer("x_min", torch.tensor(x_min), persistent=False)
        self.register_buffer("x_max", torch.tensor(x_max), persistent=False)
        self.register_buffer("y_min", torch.tensor(y_min), persistent=False)
        self.register_buffer("y_max", torch.tensor(y_max), persistent=False)
        self.register_buffer(
            "dx",
            torch.tensor((x_max - x_min) / num_segments),
            persistent=False,
        )

    # ----------------------------------------------------------------------------- #
    #                                 forward / inverse                             #
    # ----------------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
        """
        *inverse=False*  : f(x)  maps **input → output**  (piece-wise linear).
        *inverse=True*   : f⁻¹(y) maps **output → input** (analytic inverse).

        Shapes are preserved; all operations export cleanly to ONNX.
        """
        # ------------------------------------------------------------------ commons
        delta = torch.nn.functional.softplus(self.raw_delta)  # (n,)  — positive increments
        denom = torch.sum((self.x_max - self.breakpoints) * delta) + 1e-10
        scale = (self.y_max - self.y_min) / denom                      # scalar

        if not inverse:
            # -------------------------------------------------------------- forward
            x = torch.clamp(x, self.x_min, self.x_max)
            relu_terms = torch.nn.functional.relu(x.unsqueeze(-1) - self.breakpoints)     # (..., n)
            raw = torch.sum(relu_terms * delta, dim=-1)                 # (...)
            return self.y_min + scale * raw

        # ---------------------------------------------------------------  inverse
        y = torch.clamp(x, self.y_min, self.y_max)          # “x” is really y here

        slopes = torch.cumsum(delta, dim=0)                 # (n,) slope per seg
        dy_seg = scale * slopes * self.dx                   # (n,) height / seg

        # y_k  = value of f at the *start* of segment k
        y_starts = self.y_min + torch.cat(
            (torch.zeros(1, device=x.device, dtype=x.dtype),
             torch.cumsum(dy_seg[:-1], dim=0))
        )                                                   # (n,)

        # locate the segment:  idx = max{k | y ≥ y_starts[k]}
        idx = (y.unsqueeze(-1) >= y_starts).to(y.dtype).sum(dim=-1) - 1  # (...)
        idx = idx.clamp(min=0, max=self.num_segments - 1).to(torch.long)

        # gather per-sample constants
        slope_k   = slopes[idx]           # (...)
        bp_k      = self.breakpoints[idx] # (...)
        y_start_k = y_starts[idx]         # (...)

        # linear solve inside the chosen segment
        x_rec = bp_k + (y - y_start_k) / (scale * slope_k + 1e-10)
        return x_rec


class Downsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Downsample, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=1, stride=1)
        self.norm1 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels)
        self.norm2 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels)
        self.norm3 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels)
        self.pool = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                                    padding=1, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        # x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        # x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x_prime = torch.nn.functional.gelu(x, approximate="tanh")
        # x_prime = torch.nn.functional.relu(x)
        x = self.pool(x_prime)

        return x, x_prime


class Upsample(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(Upsample, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=2*in_channels, out_channels=out_channels, kernel_size=kernel_size, # 2*in_channels
                                     padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=1, stride=1)
        self.conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     padding=1, stride=1)
        self.norm1 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels) if out_channels > 3 else None
        self.norm2 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels) if out_channels > 3 else None
        self.norm3 = torch.nn.GroupNorm(num_groups=out_channels // 16, num_channels=out_channels) if out_channels > 3 else None
        self.upscale = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels, kernel_size=4,
                                                padding=1, stride=2)

    def forward(self, x, x_conc):
        x = self.upscale(x)
        # Add skip connection to improve gradient flow
        # x = x + x_conc
        x = torch.concatenate([x, x_conc], dim=1)

        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        # x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        if self.norm2 is not None:
            x = self.norm2(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        # x = torch.nn.functional.relu(x)
        x = self.conv3(x)
        if self.norm3 is not None:
            x = self.norm3(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")  # If last layer skip norm
            # x = torch.nn.functional.relu(x)

        return x


# TODO : think if ending the decoder at 64 channels and add a 1x1 convolution later to arrive at 3 channels
class UNetUnconditioned(torch.nn.Module):
    def __init__(self, layer_sizes, image_channels=3, apply_softplus=False):
        super(UNetUnconditioned, self).__init__()

        # self.image_normalization = PiecewiseLinear(8)

        self.layer_sizes = layer_sizes
        self.down_layers = torch.nn.ModuleList()
        self.up_layers = torch.nn.ModuleList()

        self.apply_softplus = apply_softplus
        
        # Add proper weight initialization
        self.apply(self._init_weights)

        # Initialize layers
        for index in range(len(self.layer_sizes)):
            in_channels = image_channels if index == 0 else self.layer_sizes[index - 1]
            out_channels = self.layer_sizes[index]

            self.down_layers.append(Downsample(in_channels=in_channels, out_channels=out_channels))
            self.up_layers.insert(0, Upsample(in_channels=out_channels, out_channels=in_channels))

    def forward(self, x):
        residuals = []

        # x = self.image_normalization(x)

        for layer in self.down_layers:
            x, x_prime = layer(x)
            residuals.append(x_prime)

        residuals = reversed(residuals)

        for layer, residual in zip(self.up_layers, residuals):
            x = layer(x, residual)
        
        if self.apply_softplus:
            x = torch.nn.functional.softplus(x)
        
        # x = self.image_normalization(x, inverse=True)

        return x
        
    @staticmethod
    def _init_weights(m):
        """Initialize network weights properly"""
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.GroupNorm):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
