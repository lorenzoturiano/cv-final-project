import scipy.signal

import albumentations as A
import numpy as np


def convolve(image, kernel, ksize):
    pad_quantity = ksize // 2
    image = np.pad(image, [[pad_quantity, pad_quantity], [pad_quantity, pad_quantity], [0, 0]], mode="reflect")
    result = scipy.signal.fftconvolve(image, kernel, mode="same")
    
    # Handle both odd and even ksize cases correctly
    if pad_quantity > 0:
        return result[pad_quantity:-pad_quantity, pad_quantity:-pad_quantity]
    return result


def build_kernel(ksize, x_sigma, y_sigma, beta, angle):
    ax = np.arange(-ksize // 2 + 1.0, ksize // 2 + 1.0)
    x, y = np.meshgrid(ax, ax)

    x_prime = x * np.cos(angle) + y * np.sin(angle)
    y_prime = -x * np.sin(angle) + y * np.cos(angle)
    grid = np.stack([x_prime, y_prime], axis=-1)

    kernel = np.power(1 + np.square(grid[:, :, 0] / x_sigma) + np.square(grid[:, :, 1] / y_sigma), -beta)

    return kernel / kernel.sum()


class RandomSeeing(A.DualTransform):
    def __init__(self, ksize, sigma_range, beta_range, angle_range, scale_range, sx_sy_max_ratio=(0.8, 1.2), always_apply=False, p=1.0):
        super(RandomSeeing, self).__init__(p, always_apply)

        self.ksize = ksize
        self.sigma_range = sigma_range
        self.beta_range = beta_range
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.sx_sy_max_ratio = sx_sy_max_ratio

    def get_params(self):
        sx = np.random.uniform(*self.sigma_range)
        sy = np.random.uniform(self.sx_sy_max_ratio[0] * sx, self.sx_sy_max_ratio[1] * sx)
        beta = np.random.uniform(*self.beta_range)
        angle = np.deg2rad(np.random.uniform(*self.angle_range))
        noise = np.random.uniform(0.9, 1.1, size=[self.ksize, self.ksize])
        scale = np.random.uniform(*self.scale_range)

        self.scale = scale

        return {"sx": sx, "sy": sy, "beta": beta, "angle": angle, "noise": noise, "scale": scale}

    def apply(self, img, **params):
        kernel = build_kernel(self.ksize, params["sx"], params["sy"], params["beta"], params["angle"])
        kernel = kernel * params["noise"]
        kernel = kernel[:, :, None].astype(np.float32)

        self.kernel = kernel

        return convolve(img, kernel, self.ksize)

    def apply_to_mask(self, img, **params):
        s = min(params["sx"], params["sy"])
        kernel = build_kernel(self.ksize, s, s, params["beta"], 0)
        kernel = kernel * params["noise"]
        kernel = kernel[:, :, None].astype(np.float32)

        return convolve(img, kernel, self.ksize)

    def apply_to_masks(self, img, **params):
        kernel = build_kernel(self.ksize, params["sx"], params["sy"], params["beta"], params["angle"])
        kernel = kernel * params["noise"]
        kernel = kernel[:, :, None].astype(np.float32)

        return convolve(img, kernel, self.ksize)


def generate_star_map_intensity():
    """
    Generate a star field intensity map with a physically-motivated distribution,
    but with parameters and scaling adjusted to ensure more stars are visible and
    a few are very bright, as in amateur astrophotography.
    Output is a 256x256 numpy array.
    """
    num_stars = 1000
    num_stars_ood = np.random.randint(0, 4)

    x_coords = np.random.randint(0, 256, size=num_stars)
    y_coords = np.random.randint(0, 256, size=num_stars)
    x_coords_ood = np.random.randint(0, 256, size=num_stars_ood)
    y_coords_ood = np.random.randint(0, 256, size=num_stars_ood)

    masses = np.zeros(num_stars)
    for i in range(num_stars):
        r = np.random.random()
        if r < 0.80:  # ~80% low mass stars (0.08-0.5 M☉)
            masses[i] = np.random.power(1.3) * 0.42 + 0.08
        elif r < 0.95:  # ~15% intermediate mass (0.5-1.0 M☉)
            masses[i] = np.random.power(2.3) * 0.5 + 0.5
        elif r < 0.98:  # ~4% high mass (1.0-20 M☉)
            masses[i] = np.random.power(2.7) * 19 + 1.0
        else:  # ~1% very massive stars (20-150 M☉)
            masses[i] = np.random.power(2.7) * 130 + 20.0

    # Mass-Luminosity relation with different power laws
    luminosities = np.zeros_like(masses)
    mask_vlm = masses < 0.43
    mask_lm = (masses >= 0.43) & (masses < 2)
    mask_im = (masses >= 2) & (masses < 20)
    mask_vms = masses >= 20

    # Different M-L relations for different mass ranges
    luminosities[mask_vlm] = 0.23 * masses[mask_vlm] ** 2.3
    luminosities[mask_lm] = masses[mask_lm] ** 4
    luminosities[mask_im] = 1.4 * masses[mask_im] ** 3.5
    luminosities[mask_vms] = 32000 * masses[mask_vms]  # Near Eddington limit

    # Adjusted distance distribution: more nearby stars, fewer extremely faint
    # Reduce max distance to increase number of visible stars
    distances = np.random.power(2.0, size=num_stars) * 300 + 10  # 10-310 pc (was 10-1010)
    luminosities /= distances ** 2

    # Add some random scatter (0.1-0.2 mag) to simulate variability
    scatter = np.random.normal(0, 0.15, size=num_stars)

    # Convert to apparent magnitudes
    magnitudes = -2.5 * np.log10(luminosities + 1e-12) + scatter

    # Scale to desired magnitude range, but compress less
    mag_min, mag_max = -2.0, 12.0
    magnitudes = np.interp(magnitudes,
                           (magnitudes.min(), magnitudes.max()),
                           (mag_min, mag_max))

    # Convert back to intensities
    intensities = 10 ** (-0.4 * magnitudes)

    # Nonlinear stretch to boost faint stars (asinh stretch, mimics photographic response)
    intensities = np.arcsinh(intensities * 10) / 3.0

    # Add a small fraction of very bright stars
    n_bright = int(0.002 * num_stars)
    if n_bright > 0:
        bright_indices = np.random.choice(num_stars, n_bright, replace=False)
        intensities[bright_indices] = np.random.uniform(2.0, 25.0, size=n_bright)

    # OOD stars: extremely bright
    intensities_ood = np.random.uniform(8.0, 2000.0, size=num_stars_ood) if num_stars_ood > 0 else []

    stars = np.zeros([256, 256], dtype=np.float32)
    valid_idx = (x_coords < 256) & (y_coords < 256) & (x_coords >= 0) & (y_coords >= 0)
    np.add.at(stars, (y_coords[valid_idx], x_coords[valid_idx]), intensities[valid_idx])

    if num_stars_ood > 0:
        valid_idx_ood = (x_coords_ood < 256) & (y_coords_ood < 256) & (x_coords_ood >= 0) & (y_coords_ood >= 0)
        np.add.at(stars, (y_coords_ood[valid_idx_ood], x_coords_ood[valid_idx_ood]), intensities_ood)

    # Gain to simulate exposure/sky brightness, but avoid over-normalizing
    gain = np.random.uniform(1.5, 4.0)
    return stars * gain


def make_blur_transform(ksize=131, sigma_range=(0.5, 6.0), beta_range=(1.5, 5.0), angle_range=(-90, 90), scale_range=(0.0, 1.0), sx_sy_max_ratio=(0.7, 1.3)):
    return A.Compose([
        RandomSeeing(ksize=ksize, sigma_range=sigma_range, beta_range=beta_range, angle_range=angle_range, scale_range=scale_range, sx_sy_max_ratio=sx_sy_max_ratio, p=1.0),
    ], additional_targets={"image0": "image", "image1": "mask", "image2": "mask"}, is_check_shapes=False)


def make_transform_crop_rescale():
    return A.Compose([
        A.RandomResizedCrop(size=[256, 256], scale=(0.1, 1.0), interpolation=3),  # interpolation = AREA
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ], additional_targets={"image0": "image"})


def make_transform_stars():
    return A.Compose([
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(p=0.5),
    ], additional_targets={"image0": "image"})
