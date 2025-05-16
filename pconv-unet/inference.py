import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import os

from pconv_unet import PartialConvUNet

def load_model(model_path, device='cpu'):
    """Load the trained PConvUNet model"""
    model = PartialConvUNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def create_random_mask(image_size, hole_size_range=(20, 80), num_holes=3):
    """Create a random binary mask with holes"""
    mask = np.ones((image_size, image_size), dtype=np.float32)
    
    for _ in range(num_holes):
        h_size = np.random.randint(hole_size_range[0], hole_size_range[1])
        w_size = np.random.randint(hole_size_range[0], hole_size_range[1])
        h_start = np.random.randint(0, image_size - h_size)
        w_start = np.random.randint(0, image_size - w_size)
        mask[h_start:h_start + h_size, w_start:w_start + w_size] = 0
        
    return mask

def preprocess_image(image_path, target_size=256):
    """Load and preprocess image to grayscale tensor"""
    # Load image
    if image_path:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((target_size, target_size))
        img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    else:
        # Create a sample gradient image if no image provided
        img_array = np.zeros((target_size, target_size), dtype=np.float32)
        for i in range(target_size):
            for j in range(target_size):
                img_array[i, j] = (i + j) / (2 * target_size)
        
        # Add some noise for realism
        noise = np.random.randn(target_size, target_size) * 0.1
        img_array += noise
        img_array = np.clip(img_array, 0, 1).astype(np.float32)
    
    return img_array

def inpaint_image(model, image, mask, device='cpu'):
    """Perform inpainting on the image using the mask"""
    # Convert to torch tensors **and cast to float32**
    image_tensor = (
        torch.from_numpy(image)
        .unsqueeze(0).unsqueeze(0)   # [B,C,H,W]
        .float()                     #  <-- add this
        .to(device)
    )
    mask_tensor = (
        torch.from_numpy(mask)
        .unsqueeze(0).unsqueeze(0)
        .float()                     #  <-- and this
        .to(device)
    )

    corrupted_tensor = image_tensor * mask_tensor      # stays float32
    with torch.no_grad():
        output = model(corrupted_tensor, mask_tensor)

    return output.squeeze().cpu().numpy()

def visualize_results(original, mask, corrupted, inpainted, save_path=None):
    """Visualize and optionally save the inpainting results"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Plot mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')
    
    # Plot corrupted
    axes[2].imshow(corrupted, cmap='gray')
    axes[2].set_title('Corrupted')
    axes[2].axis('off')
    
    # Plot inpainted
    axes[3].imshow(inpainted, cmap='gray')
    axes[3].set_title('Inpainted')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Results saved to {save_path}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inpaint images using trained PConvUNet')
    parser.add_argument('--model', type=str, default='pconv_unet.pth', help='Path to trained model')
    parser.add_argument('--image', type=str, default=None, help='Path to input image (or None for synthetic)')
    parser.add_argument('--output', type=str, default='inpainted_result.png', help='Path to save output visualization')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to use')
    args = parser.parse_args()
    
    # Check CUDA availability if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load model
    model = load_model(args.model, args.device)
    print(f"Model loaded from {args.model}")
    
    # Load/create image
    image = preprocess_image(args.image)
    print("Image preprocessed")
    
    # Create mask
    mask = create_random_mask(image.shape[0])
    print("Mask created")
    
    # Create corrupted image
    corrupted = image * mask
    
    # Inpaint
    inpainted = inpaint_image(model, image, mask, args.device)
    print("Inpainting completed")
    
    # Visualize and save
    visualize_results(image, mask, corrupted, inpainted, args.output)

if __name__ == "__main__":
    main()