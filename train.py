import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.utils import *
import datetime

import yaml

USE_WANDB = False
if not USE_WANDB:
    os.environ["WANDB_MODE"] = "disabled"
import wandb

from models.unet_convnext_attention import UNetConvNeXtAttention


from evaluate import evaluate
from models.unet import UNet
from models.unet_residual import UNetResidual
from models.unet_attention import UNetResidualAttention
from utils.data_loading import ISIC2018Task2
from utils.dice_score import *

train_dir_img = Path('./isic2018_resized/train/ISIC2018_Task1-2_Training_Input/')
train_dir_mask = Path('./isic2018_resized/train/ISIC2018_Task2_Training_GroundTruth_v3/')

val_dir_img = Path('./isic2018_resized/val/ISIC2018_Task1-2_Validation_Input/')
val_dir_mask = Path('./isic2018_resized/val/ISIC2018_Task2_Validation_GroundTruth/')

dir_checkpoint = Path('./checkpoints/')
dir_images = Path('./images/')



# Load config from YAML file
def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
    model_name='',
    scheduler_selector=None,
    scheduler_config=None,
    optimizer_selector=None,
    optimizer_config=None,
    eval_every_epochs: int = 1,
):
    
    # input_size_h = 256
    # input_size_w = 256

    # For tracking metrics over validation steps
    dice_scores_val = []
    dice_scores_train = []
    jaccard_scores_val = []
    jaccard_scores_train = []

    val_transformer = PairCompose([
      ToTensorPair(),
      MyNormalize(mean=[0.61788, 0.49051, 0.43048],
                  std=[0.19839, 0.16931, 0.16544]),
      ])
    train_transformer = PairCompose([
        ToTensorPair(),                      # Convert to tensor first
        MyNormalize(mean=[0.61788, 0.49051, 0.43048], # Pre-calculated from ISIC dataset
                std=[0.19839, 0.16931, 0.16544]),
        RandomHorizontalFlipPair(p=0.5),
        RandomVerticalFlipPair(p=0.5),
    ])

    # 1. Create dataset
    try:
        train_set = ISIC2018Task2(train_dir_img, train_dir_mask,transform=train_transformer) # TODO add preprocess
        val_set = ISIC2018Task2(val_dir_img, val_dir_mask,transform=val_transformer)
    except (AssertionError, RuntimeError, IndexError):
        logging.error("failed initializing ISIC2018Task2")

    # train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    n_train = len(train_set)
    n_val = len(val_set)
    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if optimizer_selector:
        optimizer = optimizer_selector(model, optimizer_config)
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    if scheduler_selector:
        scheduler = scheduler_selector(optimizer, scheduler_config)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

#=============================================== WEIGHTED LOSS =========================
    class_frequencies = torch.tensor([
    0.492,  # pigment_network
    0.061,  # negative_network
    0.032,  # streaks
    0.22,  # milia_like_cyst
    0.195,  # globules
    ])

    # --- 2. Calculate the pos_weight tensor using inverse frequency ---
    # We add a small epsilon to avoid division by zero if a frequency is 0
    epsilon = 1e-6
    inverse_freq_weights = 1.0 / (class_frequencies + epsilon)

    # --- (Optional but Recommended) Normalize the weights ---
    # Normalizing can lead to more stable training. A common way is to 
    # divide by the smallest weight, making the smallest weight 1.0.
    normalized_weights = inverse_freq_weights / inverse_freq_weights.min()
    # pos_weights_tensor = normalized_weights.to(device)
    pos_weights_tensor = normalized_weights.view(1, -1, 1, 1).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
    # criterion = nn.BCEWithLogitsLoss()
#========================================================================================

    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # print("masks_pred shape:", masks_pred.shape)
                    # print("true_masks shape:", true_masks.shape)
                    
                    # bce_loss = criterion(masks_pred, true_masks)
                    # loss = bce_loss + dice_loss(torch.sigmoid(masks_pred), true_masks)
                    loss = dice_loss(torch.sigmoid(masks_pred), true_masks)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

        # Evaluation at the end of each epoch if required
        if epoch % eval_every_epochs == 0:
            dice_val_score, jaccard_val_score = evaluate(model, val_loader, device, amp)
            dice_train_score, jaccard_train_score = evaluate(model, train_loader, device, amp)
            dice_scores_val.append(float(dice_val_score))
            jaccard_scores_val.append(float(jaccard_val_score))
            dice_scores_train.append(float(dice_train_score))
            jaccard_scores_train.append(float(jaccard_train_score))
            sched_type = scheduler.__class__.__name__
            if sched_type == "ReduceLROnPlateau":
                scheduler.step(dice_val_score)
            else:
                scheduler.step()
            tqdm.write(f'Validation Dice score: {dice_val_score}')
            tqdm.write(f'Validation jaccard_index: {jaccard_val_score}')


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # Format: YYYY-MM-DD_HH-MM-SS_modelname_epochX.pth
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_name = f"{now}_{model_name}_epoch{epoch}.pth"
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / checkpoint_name))
            logging.info(f"Checkpoint {checkpoint_name} saved!")

    import matplotlib.pyplot as plt

    # Gather scheduler, optimizer, and learning rate information
    scheduler_type = None
    optimizer_type = None
    if scheduler_config is not None:
        scheduler_type = scheduler_config.get("scheduler", "unknown")
    else:
        scheduler_type = "unknown"
    if optimizer_config is not None:
        optimizer_type = optimizer_config.get("optimizer", "unknown")
    else:
        optimizer_type = "unknown"

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    val_steps = range(1, len(dice_scores_val) + 1)

    # Dice subplot
    axes[0].plot(val_steps, dice_scores_val, marker='o', label='Validation Dice')
    axes[0].plot(val_steps, dice_scores_train, marker='x', label='Train Dice')
    axes[0].set_xlabel("Validation Step")
    axes[0].set_ylabel("Dice Score")
    axes[0].set_title(
        f"Dice Score vs Step\nModel={model_name}, Scheduler={scheduler_type}, "
        f"Optimizer={optimizer_type}, LR={learning_rate}, Epochs={epochs}"
    )
    axes[0].legend()
    axes[0].grid(True)

    # Jaccard subplot
    axes[1].plot(val_steps, jaccard_scores_val, marker='o', label='Validation Jaccard')
    axes[1].plot(val_steps, jaccard_scores_train, marker='x', label='Train Jaccard')
    axes[1].set_xlabel("Validation Step")
    axes[1].set_ylabel("Jaccard Index")
    axes[1].set_title(
        f"Jaccard Index vs Step\nModel={model_name}, Scheduler={scheduler_type}, "
        f"Optimizer={optimizer_type}, LR={learning_rate}, Epochs={epochs}"
    )
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    plot_name = (
        f"dice_jaccard_vs_step_{model_name}_model-{model_name}_"
        f"sched-{scheduler_type}_opt-{optimizer_type}_lr-{learning_rate}_ep{epochs}.png"
    )
    fig.savefig(str(dir_images / plot_name))
    plt.show()


# --- Optimizer selection ---
def select_optimizer(model, config):
    opt_type = config.get("optimizer", "adam").lower()
    lr = float(config.get("lr", 1e-4))
    wd = float(config.get("weight_decay", 0.0))
    momentum = float(config.get("momentum", 0.9))
    if opt_type == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif opt_type == "rmsprop":
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif opt_type == "adagrad":
        return optim.Adagrad(model.parameters(), lr=lr, weight_decay=wd)
    # elif opt_type == "adabelief":
    #     from adabelief_pytorch import AdaBelief
    #     return AdaBelief(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


# --- Scheduler selection inside train_model ---
def select_scheduler(optimizer, config):
    sched_type = config.get("scheduler", "reduce_lr")
    if sched_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", int(config.get("epochs", 40)))),
            eta_min=float(config.get("eta_min", 1e-6))
        )
    elif sched_type == "cosine_restart":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config.get("t_0", 10)),
            T_mult=int(config.get("t_mult", 2)),
            eta_min=float(config.get("eta_min", 1e-6))
        )
    elif sched_type == "reduce_lr":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.get("plateau_mode", "max"),
            patience=int(config.get("patience", 3)),
            factor=float(config.get("factor", 0.5))
        )
    elif sched_type == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 10)),
            gamma=float(config.get("gamma", 0.1))
        )
    elif sched_type == "exp":
        return optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=float(config.get("gamma", 0.95))
        )
    elif sched_type == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config.get("max_lr", 1e-3)),
            steps_per_epoch=int(config.get("steps_per_epoch", 100)),
            epochs=int(config.get("epochs", 40))
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# --- Weight initialization utility ---
def initialize_weights(model, method="none", exclude=None):
    if method == "none":
        return  # Use default initialization
    for m in model.modules():
        # If m is within the excluded module (e.g., model.encoder), skip it
        if exclude is not None:
            # Check if m is part of exclude (works for nn.Module or list of nn.Module)
            if isinstance(exclude, nn.Module):
                if any(m is mod for mod in exclude.modules()):
                    continue
            elif isinstance(exclude, (list, tuple)):
                if any(any(m is mod for mod in exc.modules()) for exc in exclude):
                    continue
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if method == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method == "kaiming":
                nn.init.kaiming_normal_(m.weight)
            elif method == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif method == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias, 0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a UNet model with YAML configuration.")
    parser.add_argument("--yaml", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    yaml_path = args.yaml
    config = load_config(yaml_path)

    # --- Device selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logging.info(f'Using device {device}')

    # --- Model selection ---
    model_name = config.get("model", "unet") # if model not defined puts unet as default
    n_classes = int(config["classes"])
    bilinear = config["bilinear"]
    if model_name == 'unet':
        model = UNet(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    elif model_name == 'unet-residual':
        model = UNetResidual(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    elif model_name == 'unet-attention':
        model = UNetResidualAttention(n_channels=3, n_classes=n_classes, bilinear=bilinear)
    elif model_name == 'unet-convnext-attention':
        encoder_type = config.get("encoder", "convnext_tiny")
        pretrained = bool(config.get("pretrained", True))
        # You can parametrize decoder_attention, here we set it to True by default
        model = UNetConvNeXtAttention(
            n_channels=3,        # or 5 if your input is 5-channel!
            n_classes=n_classes,
            bilinear=True,       # or False if you want transposed conv
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # --- Weight initialization ---
    init_method = config.get("init", "none")
    if model_name == 'unet-convnext-attention' and hasattr(model, "encoder"):
        initialize_weights(model, method=init_method, exclude=model.encoder)
    else:
        initialize_weights(model, method=init_method)

    model = model.to(memory_format=torch.channels_last).to(device)
    
    # --- Optional: load weights ---
    if config.get("load"):
        state_dict = torch.load(config["load"], map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {config["load"]}')

    # --- Training ---
    train_model(
        model=model,
        device=device,
        epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["lr"]),
        img_scale=float(config["scale"]),
        val_percent=float(config["val"]) / 100,
        amp=bool(config["amp"]),
        weight_decay=float(config.get("weight_decay", 1e-8)),
        momentum=float(config.get("momentum", 0.999)),
        model_name=model_name,
        # pass config or scheduler selector as needed
        scheduler_selector=select_scheduler,
        scheduler_config=config,
        optimizer_selector=select_optimizer,
        optimizer_config=config,
        eval_every_epochs=int(config.get("eval_every_epochs", 1)),
    )


'''

            globules    milia
globules                
mila


'''