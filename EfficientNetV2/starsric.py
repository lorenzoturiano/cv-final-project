import os
import random
from typing import Tuple, Dict, Optional

import scipy.signal

import torch
import numpy as np
import os
from PIL import Image
import torch.utils.data
import torchvision.transforms.functional as TF




def log_norm(x, min_=None, mean_=None, std_=None, epsilon=1e-3, clip=False):
    min_ = torch.min(x.reshape(x.shape[0], -1), dim=1).values[:, None, None] if min_ is None else min_

    x = x - min_ + epsilon
    if clip:
        x = torch.clip(x, epsilon, torch.inf)

    x = torch.log(x)

    mean_ = torch.mean(x.reshape(x.shape[0], -1), dim=1)[:, None, None] if mean_ is None else mean_
    std_ = torch.std(x.reshape(x.shape[0], -1), dim=1)[:, None, None] if std_ is None else std_
    # Prevent division by zero
    std_ = torch.clamp(std_, min=epsilon)

    x = (x - mean_) / std_ * 0.1
    x = torch.nan_to_num(x, 0)

    return x, min_, mean_, std_


class StarRemovalDataset(torch.utils.data.Dataset):
    """Dataset for training deconvolution models with realistic astronomical images."""
    
    def __init__(self, 
                 image_folder: str, 
                 psf_folder: str, 
                 image_transform, 
                 blur_transform, 
                 star_generator, 
                 seed: Optional[int] = None):
        """
        Initialize the deconvolution aberration dataset.
        
        Args:
            image_folder: Path to folder containing background images (.npy files)
            psf_folder: Path to folder containing PSF pairs (.npy files)
            image_transform: Albumentations transform for input images
            blur_transform: Transform for applying atmospheric seeing effects
            star_generator: Function to generate realistic star field
            seed: Random seed for reproducibility
        """
        super(StarRemovalDataset, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # Validate directory existence
        if not os.path.exists(image_folder):
            raise ValueError(f"Image folder does not exist: {image_folder}")
        if not os.path.exists(psf_folder):
            raise ValueError(f"PSF folder does not exist: {psf_folder}")
            
        # Get all valid image and PSF files
        self.image_files = [os.path.join(image_folder, file) 
                           for file in os.listdir(image_folder) 
                           if file.endswith(".npy") and not file.startswith(".")]
        self.psf_files = [os.path.join(psf_folder, file) 
                         for file in os.listdir(psf_folder) 
                         if file.endswith(".npy") and not file.startswith(".")]
        
        # Validate that files were found
        if not self.image_files:
            raise ValueError(f"No .npy files found in image folder: {image_folder}")
        if not self.psf_files:
            raise ValueError(f"No .npy files found in PSF folder: {psf_folder}")

        self.image_transform = image_transform
        self.blur_transform = blur_transform
        self.star_generator = star_generator

    def __len__(self) -> int:
        return 32_768
    
    def generate_noise(self, img, gain_e_per_adu=10_000.0, read_rms_e=10.0, drizzle_noise=True, kernel_sigma=0.5, kernel_size=7):
        kernel = np.outer(scipy.signal.windows.gaussian(kernel_size, kernel_sigma), scipy.signal.windows.gaussian(kernel_size, kernel_sigma))
        kernel /= np.sum(kernel)
        kernel = kernel[:, :, np.newaxis]

        signal_e = img * gain_e_per_adu
        var_e = signal_e + read_rms_e**2
        # Prevent negative variances due to negative pixel values
        var_e = np.clip(var_e, 0, None)
        sigma_map = np.sqrt(var_e) / gain_e_per_adu

        noise = np.random.normal(size=img.shape)

        if drizzle_noise:
            noise = scipy.signal.oaconvolve(noise, kernel, mode="same")
            noise /= noise.std(dtype=np.float32)
        
        return noise * sigma_map
    
    def background_rescale(self, img, img_scale_range=(0.6, 1.2)):
        safe_shift_min = -np.percentile(img, 1)

        img_scale = np.random.uniform(img_scale_range[0], img_scale_range[1])

        max_shift = min(0.1, safe_shift_min * 0.8)
        img_shift = np.random.uniform(-max_shift, max_shift)

        img = img * img_scale + img_shift

        return img
 
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Randomly select files (independent of the idx)
        image_file = np.random.choice(self.image_files)
        psf_file = np.random.choice(self.psf_files)

        # Loading files lazily with memmap to preserve memory and time
        image = np.load(image_file, mmap_mode="c")
        psf_pair = np.load(psf_file).astype(np.float32)
                
        # Extract PSF components
        psf = psf_pair[np.random.choice([0, 1]), :, :, np.newaxis]
        
        # Normalize PSFs to ensure they sum to 1 (should be redundant)
        psf = psf / np.sum(psf)
            
        # Preparing data
        transformed = self.image_transform(image=image)
        image = transformed["image"]
        image = self.background_rescale(image)
        star_map = self.star_generator()[:, :, np.newaxis].astype(np.float32)

        # Apply diffraction (convolve background image and star map with PSFs)
        image = scipy.signal.fftconvolve(image, psf, mode="same")
        star_map = scipy.signal.fftconvolve(star_map, psf, mode="same")

        # Apply atmospheric seeing effects (blur transform)
        transformed = self.blur_transform(image=image, image0=star_map)
        image = transformed["image"]
        star_map = transformed["image0"]

        noise = self.generate_noise(image, drizzle_noise=bool(random.getrandbits(1)), read_rms_e=np.random.uniform(0.0, 15.0))
        image = image + noise

        # Apply stars clipping
        star_map = np.clip(star_map, 0.0, np.random.uniform(0.1, 1.0))

        # Add stars
        image_with_stars = image + star_map

        image_with_stars = np.clip(image_with_stars, 0.0, 1.0)
        image = np.clip(image, 0.0, 1.0)

        # Transform all components to PyTorch tensors
        image_with_stars = torch.from_numpy(image_with_stars.astype(np.float32)).permute(2, 0, 1)
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)

        # Normalize data 
        image_with_stars, min_, mean_, std_ = log_norm(image_with_stars, epsilon=1e-3)
        image, _, _, _ = log_norm(image, min_=min_, mean_=mean_, std_=std_, epsilon=1e-3)

        return image_with_stars, image, star_map
    

import os
import torch
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from typing import Union
from tqdm import tqdm

    def save_dataset_samples(
        dataset: torch.utils.data.Dataset,
        output_dir: str = "output",
        num_samples: int = 10,
        device: Union[str, torch.device] = "cpu"
    ):
        # Crea le directory se non esistono
        backgrounds_dir = os.path.join(output_dir, "backgrounds")
        images_dir = os.path.join(output_dir, "images")
        masks_dir = os.path.join(output_dir, "masks")
        os.makedirs(backgrounds_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        for i in tqdm(range(num_samples), desc="Saving samples"):
            image_with_stars, image, star_map = dataset[i]

            # Sposta su device se necessario
            image_with_stars = image_with_stars.to(device)
            image = image.to(device)
            star_map = torch.from_numpy(star_map).permute(2, 0, 1).to(device)

            # Normalizza da 0 a 1 per salvataggio
            def normalize(x):
                x = x - x.min()
                if x.max() > 0:
                    x = x / x.max()
                return x

            image_with_stars = normalize(image_with_stars)
            image = normalize(image)
            star_map = normalize(star_map)

            # Salva immagini
            save_image(image_with_stars, os.path.join(images_dir, f"{i:03d}.png"))
            save_image(image, os.path.join(backgrounds_dir, f"{i:03d}.png"))
            save_image(star_map, os.path.join(masks_dir, f"{i:03d}.png"))

        print(f"✅ Salvati {num_samples} esempi in '{output_dir}'")




if __name__ == "__main__":

    
    dataset = StarRemovalDataset(
        image_folder="smallreal/backgrounds",
        psf_folder="smallreal/stretched",
        image_transform=None,
        blur_transform=None,
        star_generator=lambda: np.random.rand(384, 384),
        seed=42
    )

    # Salva i primi 20 esempi
    save_dataset_samples(dataset, output_dir="output_samples", num_samples=5)
