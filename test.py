import argparse
import torch
from torch import Tensor

from pathlib import Path
import yaml

from utils.data_loading import ISIC2018Task2
from utils.score_loss_functions import dice_coeff, jaccard_index
from models.unet import UNet
from models.unet_residual import UNetResidual
from models.unet_attention import UNetResidualAttention
from models.unet_convnext_attention import UNetConvNeXtAttention

# --- For consistent transforms ---
from train import PairCompose, ToTensorPair, MyNormalize


#COMMAND: python test.py --checkpoint images/top3_convnext_regular_loss/2025-07-05_15-37-31_unet-convnext-attention_epoch23_best_0.2052_jaccard_reducelr.pth --yaml model_params/unet_convnext_configs/config1_unet_convnext.yaml

def dice_coeff_test(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Computes the Dice coefficient for multi-label segmentation.
    Expects input of shape (N, C, H, W).
    Computes the Dice score for each class and averages them.
    """
    assert input.size() == target.size()
    assert input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"

    # Sum over batch and spatial dimensions (N, H, W) for each class
    inter = (input * target).sum(dim=(0, 2, 3))
    sets_sum = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))

    # Handle cases where a class is not present in either input or target
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (2 * inter + epsilon) / (sets_sum + epsilon)
    return dice.mean(), dice # Average Dice score across all classes



def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_model(model_type, n_channels, n_classes, bilinear, device, checkpoint_path):
    if model_type == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_type == 'unet-residual':
        model = UNetResidual(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_type == 'unet-attention':
        model = UNetResidualAttention(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    elif model_type == 'unet-convnext-attention':
        model = UNetConvNeXtAttention(
              n_channels=3,        # or 5 if your input is 5-channel!
              n_classes=n_classes,
              bilinear=True,       # or False if you want transposed conv
              pretrained=False,
              freeze_encoder=False  # or False
          )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    model = model.to(memory_format=torch.channels_last).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Make sure you use the same model config as in training!")
        exit(1)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Test a segmentation model on the ISIC test set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--yaml", type=str, required=True, help="Path to YAML config used for model params")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides YAML)")
    parser.add_argument("--gpu", type=int, default=None, help="GPU id to use (default: auto)")
    args = parser.parse_args()

    config = load_config(args.yaml)
    # Override batch size if given
    batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 16)
    model_name = config.get("model", "unet")
    bilinear = config.get("bilinear", True)
    n_classes = config.get("classes", 5)

    # --- Device selection ---
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # --- Data loading ---
    test_dir_img = Path('./isic2018_resized/test/ISIC2018_Task1-2_Test_Input/')
    test_dir_mask = Path('./isic2018_resized/test/ISIC2018_Task2_Test_GroundTruth/')
    # test_dir_img = Path('./isic2018_resized/val/ISIC2018_Task1-2_Validation_Input/')
    # test_dir_mask = Path('./isic2018_resized/val/ISIC2018_Task2_Validation_GroundTruth/')
    test_transform = PairCompose([
        ToTensorPair(),
        MyNormalize(mean=[0.61788, 0.49051, 0.43048], std=[0.19839, 0.16931, 0.16544]),
    ])
    test_set = ISIC2018Task2(test_dir_img, test_dir_mask, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- Model loading ---
    model = load_model(model_name, n_channels=3, n_classes=n_classes, bilinear=bilinear, device=device, checkpoint_path=args.checkpoint)

    # --- Evaluation (batch-size invariant, global metrics) ---
    all_preds = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = batch['mask'].to(device, dtype=torch.float32)
            masks_pred = torch.sigmoid(model(images))
            all_preds.append(masks_pred.cpu())
            all_targets.append(masks_true.cpu())

    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    dice_score_mean, dice_score_vec = dice_coeff_test(all_preds, all_targets)
    jac_score = jaccard_index(all_preds, all_targets)

    print("dice_score_vec: ", dice_score_vec)
    print("Evaluation complete!")
    print(f"Global Dice score:    {dice_score_mean:.4f}")
    print(f"Global Jaccard index: {jac_score:.4f}")

if __name__ == "__main__":
    main()