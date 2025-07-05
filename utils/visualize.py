import argparse
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt


# --- Important: Make sure these imports match your project structure ---
from utils.data_loading import ISIC2018Task2
from models.unet import UNet
from models.unet_residual import UNetResidual
from models.unet_attention import UNetResidualAttention
from train import PairCompose, ToTensorPair, MyNormalize # Using MyNormalize from train.py

# --- Define class names based on train.py for plot titles ---
CLASS_NAMES = [
    'Pigment Network',
    'Negative Network',
    'Streaks',
    'Milia-like Cyst',
    'Globules'
]

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_type, n_channels, n_classes, bilinear, device, checkpoint_path):
    # This function is copied from your test.py
    if model_type == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_type == 'unet-residual':
        model = UNetResidual(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_type == 'unet-attention':
        model = UNetResidualAttention(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model = model.to(memory_format=torch.channels_last).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def unnormalize(tensor, mean, std):
    """Reverses the normalization on a tensor image."""
    tensor = tensor.clone() # Avoid modifying the original tensor
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def main():
    parser = argparse.ArgumentParser(description="Visualize a segmentation model's binary output for each class")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML config used for model params")
    parser.add_argument("--image-index", type=int, default=10, help="Index of the image in the test set to visualize")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to use (default: auto)")
    parser.add_argument("--output-file", type=str, default="visualization_binary.png", help="Path to save the output plot")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold to create binary masks")
    parser.add_argument("--train", type=int, default=0, help="use an image for train, put 1")

    args = parser.parse_args()

    config = load_config(args.yaml)
    model_name = config.get("model", "unet")
    bilinear = config.get("bilinear", True)
    n_classes = config.get("classes", 5)
    train = int(args.train)
    
    if n_classes != len(CLASS_NAMES):
        raise ValueError(f"Number of classes in config ({n_classes}) does not match CLASS_NAMES ({len(CLASS_NAMES)})")

    # --- Device selection ---
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data loading for a SINGLE image ---
    test_dir_image = Path('./isic2018_resized/test/ISIC2018_Task1-2_Test_Input')
    test_dir_mask = Path('./isic2018_resized/test/ISIC2018_Task2_Test_GroundTruth')

    if train == 1:
        test_dir_image = Path('./isic2018_resized/train/ISIC2018_Task1-2_Training_Input')
        test_dir_mask = Path('./isic2018_resized/train/ISIC2018_Task2_Training_GroundTruth_v3')
    
    
    norm_mean = [0.61788, 0.49051, 0.43048]
    norm_std = [0.19839, 0.16931, 0.16544]
    
    test_transform = PairCompose([
        ToTensorPair(),
        MyNormalize(mean=norm_mean, std=norm_std),
    ])
    test_set = ISIC2018Task2(test_dir_image, test_dir_mask, transform=test_transform)
    
    sample = test_set[args.image_index]
    image_tensor = sample['image']
    true_mask_tensor = sample['mask'] # Shape: [C, H, W]

    # --- Model loading ---
    model = load_model(model_name, n_channels=3, n_classes=n_classes, bilinear=bilinear, device=device, checkpoint_path=args.checkpoint)

    # --- Inference on the single image ---
    input_batch = image_tensor.unsqueeze(0).to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
    
    with torch.no_grad():
        output_logits = model(input_batch)
        # Apply sigmoid to get probabilities [0, 1] for each class
        pred_probs = torch.sigmoid(output_logits)

    # --- Process Tensors for Plotting ---
    # Create binary masks by applying the threshold
    pred_binary_masks = (pred_probs > args.threshold).float() # Shape: [1, C, H, W]
    
    # Remove batch dimension for plotting
    pred_binary_masks = pred_binary_masks.squeeze(0) # Shape: [C, H, W]

    # Un-normalize the original image for display
    display_image = unnormalize(image_tensor, mean=norm_mean, std=norm_std)
    display_image = display_image.permute(1, 2, 0) # Permute to [H, W, C] for matplotlib

    # --- Plotting ---
    # Create a figure with 3 rows and n_classes columns
    fig, axes = plt.subplots(3, n_classes, figsize=(20, 8))
    fig.suptitle(f'Model: {model_name} | Image Index: {args.image_index} | Threshold: {args.threshold}', fontsize=16)

    # --- Plot Original Image spanning the top row ---
    # We turn off all axes in the first row and plot the image in the first one
    for i in range(n_classes):
        axes[0, i].axis('off')
    # Plot the main image in the first subplot of the first row
    axes[0, 0].imshow(display_image)
    axes[0, 0].set_title('Original Image', fontsize=14)

    # --- Plot Ground Truth and Predicted Masks ---
    for i in range(n_classes):
        # Plot Ground Truth
        ax_gt = axes[1, i]
        gt_mask_i = true_mask_tensor[i].cpu()
        ax_gt.imshow(gt_mask_i, cmap='gray')
        ax_gt.set_title(CLASS_NAMES[i], fontsize=12)
        ax_gt.set_xticks([])
        ax_gt.set_yticks([])

        # Plot Prediction
        ax_pred = axes[2, i]
        pred_mask_i = pred_binary_masks[i].cpu()
        ax_pred.imshow(pred_mask_i, cmap='gray')
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

    # Set row labels
    axes[1, 0].set_ylabel('Ground Truth', fontsize=14, labelpad=10)
    axes[2, 0].set_ylabel('Prediction', fontsize=14, labelpad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save and show the figure
    plt.savefig(args.output_file, bbox_inches='tight', dpi=150)
    print(f"Binary visualization saved to {args.output_file}")
    plt.show()

if __name__ == "__main__":
    main()