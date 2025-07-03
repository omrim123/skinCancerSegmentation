import argparse
import torch
from pathlib import Path
import yaml

from utils.data_loading import ISIC2018Task2
from utils.dice_score import dice_coeff, jaccard_index
from models.unet import UNet
from models.unet_residual import UNetResidual
from models.unet_attention import UNetResidualAttention

# --- For consistent transforms ---
from train import PairCompose, ToTensorPair, MyNormalize


#COMMAND: python test.py --checkpoint checkpoints/2025-07-03_17-44-50_unet-attention_epoch9.pth --yaml model_params/config1.yaml


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

    # --- Evaluation ---
    dice_total, jaccard_total, num_batches = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = batch['mask'].to(device, dtype=torch.float32)
            masks_pred = torch.sigmoid(model(images))
            dice_score = dice_coeff(masks_pred, masks_true)
            jac_score = jaccard_index(masks_pred, masks_true)
            dice_total += dice_score.item()
            jaccard_total += jac_score.item()
            num_batches += 1
    print("Evaluation complete!")
    print(f"Average Dice score:    {dice_total / num_batches:.4f}")
    print(f"Average Jaccard index: {jaccard_total / num_batches:.4f}")

if __name__ == "__main__":
    main()