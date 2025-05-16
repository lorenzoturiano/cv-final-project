import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class InpaintingDataset(Dataset):
    """
    Dataset for inpainting task that generates synthetic data for demonstration
    """
    def __init__(self, num_samples=1000, image_size=256):
        self.num_samples = num_samples
        self.image_size = image_size
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Create a synthetic grayscale image (e.g., with gradients or patterns)
        image = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        
        # Create a simple gradient pattern
        for i in range(self.image_size):
            for j in range(self.image_size):
                image[i, j] = (i + j) / (2 * self.image_size)
                
        # Add some noise or patterns for diversity
        noise = np.random.randn(self.image_size, self.image_size) * 0.1
        image += noise
        image = np.clip(image, 0, 1).astype(np.float32)
        
        # Create a random binary mask (1 for valid pixels, 0 for holes)
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)
        
        # Add random holes
        num_holes = np.random.randint(1, 5)
        for _ in range(num_holes):
            h_size = np.random.randint(20, 80)
            w_size = np.random.randint(20, 80)
            h_start = np.random.randint(0, self.image_size - h_size)
            w_start = np.random.randint(0, self.image_size - w_size)
            mask[h_start:h_start + h_size, w_start:w_start + w_size] = 0
            
        # Create corrupted image (set holes to 0)
        corrupted_image = image * mask
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        corrupted_tensor = torch.from_numpy(corrupted_image).unsqueeze(0)
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'corrupted': corrupted_tensor
        }


def train_model(model, dataloader, num_epochs=10, device='cuda'):
    """Train the inpainting model"""
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
    else:
        device = 'cpu'
        
    # Define loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Get data
            original = batch['image'].to(device)
            mask = batch['mask'].to(device)
            corrupted = batch['corrupted'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(corrupted, mask)
            
            # Calculate loss (only for the masked regions)
            loss = criterion(output * (1 - mask), original * (1 - mask))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model


def visualize_results(model, dataset, num_samples=3, device='cpu'):
    """Visualize inpainting results"""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    for i in range(num_samples):
        # Get a random sample
        sample_idx = np.random.randint(0, len(dataset))
        sample = dataset[sample_idx]
        
        # Get tensors
        original = sample['image'].unsqueeze(0).to(device)
        mask = sample['mask'].unsqueeze(0).to(device)
        corrupted = sample['corrupted'].unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(corrupted, mask)
        
        # Convert tensors to numpy for visualization
        original_np = original.squeeze().cpu().numpy()
        corrupted_np = corrupted.squeeze().cpu().numpy()
        output_np = output.squeeze().cpu().numpy()
        
        # Display
        axes[i, 0].imshow(original_np, cmap='gray')
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(corrupted_np, cmap='gray')
        axes[i, 1].set_title('Corrupted (with holes)')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(output_np, cmap='gray')
        axes[i, 2].set_title('Inpainted')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


# Main execution code
def main():
    from pconv_unet import PartialConvUNet
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    dataset = InpaintingDataset(num_samples=500, image_size=256)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    
    # Create model
    model = PartialConvUNet()
    
    # Train the model
    print("Starting training...")
    model = train_model(model, dataloader, num_epochs=5, device=device)
    
    # Save the model
    torch.save(model.state_dict(), 'pconv_unet.pth')
    print("Model saved to pconv_unet.pth")
    
    # Visualize results
    visualize_results(model, dataset, num_samples=3, device=device)


if __name__ == "__main__":
    main()