import os
import torch
import matplotlib.pyplot as plt


@torch.no_grad()
def plot_batch_v2(image, pred, gt, epoch, num_of_images=8, folder="./", b=None):
    # Determine number of columns based on whether b is provided
    num_cols = 4 if b is not None else 3
    fig, axs = plt.subplots(num_of_images, num_cols, figsize=(16.5 * num_cols/3, 30.5))

    for i in range(num_of_images):
        axs[i, 0].imshow(image[i].cpu().permute(1, 2, 0), interpolation="lanczos", cmap="grey")
        axs[i, 0].set_title("Blurred Input")
        
        axs[i, 1].imshow(pred[i].cpu().permute(1, 2, 0), interpolation="lanczos", cmap="grey")
        axs[i, 1].set_title("Deconvolved Output")
        
        axs[i, 2].imshow(gt[i].cpu().permute(1, 2, 0), interpolation="lanczos", cmap="grey")
        axs[i, 2].set_title("Ground Truth")
        
        # If b values are provided, show them in the fourth column
        if b is not None:
            axs[i, 3].imshow(b[i].cpu().permute(1, 2, 0), interpolation="lanczos", cmap="grey")
            axs[i, 3].set_title("B Values")

    fig.savefig(os.path.join(folder, f"plot_{epoch}.png"), bbox_inches='tight')
    plt.close(fig)


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


def log_denorm(x, min_, mean_, std_, epsilon=1e-3):
    x = x * std_ / 0.1 + mean_
    x = torch.exp(x)
    x = x + min_ - epsilon

    return x
