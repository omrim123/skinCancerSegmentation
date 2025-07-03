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
):
    
    # input_size_h = 256
    # input_size_w = 256

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

                    # loss += dice_loss(
                    #     F.softmax(masks_pred, dim=1).float(),
                    #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                    #     multiclass=True
                    # )
                    
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

                # Evaluation round
                division_step = (n_train // (batch_size))
                # division_step = (n_train // (2 * batch_size)) # every 50% 
                # division_step = max(1, n_train // (40 * batch_size))  # 5% increments for tests
                if division_step > 0:
                    if global_step % division_step == 0:

                        # histograms = {}
                        # for tag, value in model.named_parameters():
                        #     tag = tag.replace('/', '.')
                        #     if not (torch.isinf(value) | torch.isnan(value)).any():
                        #         histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        #     if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        #         histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        dice_val_score, jaccard_val_score = evaluate(model, val_loader, device, amp)
                        sched_type = scheduler.__class__.__name__
                        if sched_type == "ReduceLROnPlateau":
                            scheduler.step(dice_val_score)
                        else:
                            scheduler.step()


                        logging.info('Validation Dice score: {}'.format(dice_val_score))
                        logging.info('Validation jaccard_index: {}'.format(jaccard_val_score))
                        # try:
                        #     experiment.log({
                        #         'learning rate': optimizer.param_groups[0]['lr'],
                        #         'validation Dice': dice_val_score,
                        #         'validation jaccard': jaccard_val_score,
                        #         'images': wandb.Image(images[0].cpu()),
                        #         'masks': {
                        #             'true': wandb.Image(true_masks[0].float().cpu()),
                        #             'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                        #         },
                        #         'step': global_step,
                        #         'epoch': epoch,
                        #         **histograms
                        #     })
                        # except:
                        #     pass


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            # Format: YYYY-MM-DD_HH-MM-SS_modelname_epochX.pth
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            checkpoint_name = f"{now}_{model_name}_epoch{epoch}.pth"
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / checkpoint_name))
            logging.info(f"Checkpoint {checkpoint_name} saved!")




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
    if model_name == 'unet':
        model = UNet(n_channels=3, n_classes=config["classes"], bilinear=config["bilinear"])
    elif model_name == 'unet-residual':
        model = UNetResidual(n_channels=3, n_classes=config["classes"], bilinear=config["bilinear"])
    elif model_name == 'unet-attention':
        model = UNetResidualAttention(n_channels=3, n_classes=config["classes"], bilinear=config["bilinear"])
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(memory_format=torch.channels_last).to(device)
    
    # --- Optional: load weights ---
    if config.get("load"):
        state_dict = torch.load(config["load"], map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {config["load"]}')

    # --- Scheduler selection inside train_model ---
    def select_scheduler(optimizer, config):
        sched_type = config.get("scheduler", "reduce_lr")
        if sched_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.get("t_max", config.get("epochs", 40)), 
                eta_min=config.get("eta_min", 1e-6)
            )
        elif sched_type == "cosine_restart":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.get("t_0", 10),
                T_mult=config.get("t_mult", 2),
                eta_min=config.get("eta_min", 1e-6)
            )
        elif sched_type == "reduce_lr":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=config.get("plateau_mode", "max"),
                patience=config.get("patience", 3),
                factor=config.get("factor", 0.5)
            )
        elif sched_type == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.get("step_size", 10),
                gamma=config.get("gamma", 0.1)
            )
        elif sched_type == "exp":
            return optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=config.get("gamma", 0.95)
            )
        elif sched_type == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=config.get("max_lr", 1e-3),
                steps_per_epoch=config.get("steps_per_epoch", 100),
                epochs=config.get("epochs", 40)
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_type}")

    # --- Training ---
    train_model(
        model=model,
        device=device,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["lr"],
        img_scale=config["scale"],
        val_percent=config["val"] / 100,
        amp=config["amp"],
        weight_decay=float(config.get("weight_decay", 1e-8)),
        momentum=config.get("momentum", 0.999),
        model_name=model_name,
        # pass config or scheduler selector as needed
        scheduler_selector=select_scheduler,
        scheduler_config=config,
    )



# class HyperParams():
#     def __init__(self, epochs, batch_size, lr, load, scale, val, amp, bilinear, classes, model):
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.lr = lr
#         self.load = load
#         self.scale = scale
#         self.val = val
#         self.amp = amp
#         self.bilinear = bilinear
#         self.classes = classes
#         self.model = model


# if __name__ == '__main__':
#     args = HyperParams(
#         epochs=40, 
#         batch_size=32,
#         lr=1e-4,
#         load=None,
#         #load='checkpoints/checkpoint_epoch1.pth',
#         scale=0.5, # delete  
#         val=0.2, 
#         amp=False, 
#         bilinear=True, 
#         classes=5,
#         # model='unet',
#         model='unet-attention',
#     )

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
#         device = torch.device('mps')
#     else:
#         device = torch.device('cpu')    
    
#     logging.info(f'Using device {device}')

#     # Change here to adapt to your data
#     # n_channels=3 for RGB images
#     # n_classes is the number of probabilities you want to get per pixel

#     if args.model == 'unet':
#         model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     elif args.model == 'unet-residual':
#         model = UNetResidual(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     elif args.model == 'unet-attention':
#         model = UNetResidualAttention(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     else:
#         model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    

#     model = model.to(memory_format=torch.channels_last)

#     logging.info(f'Network:\n'
#                  f'\t{model.n_channels} input channels\n'
#                  f'\t{model.n_classes} output channels (classes)\n'
#                  f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

#     if args.load:
#         state_dict = torch.load(args.load, map_location=device)
#         # del state_dict['mask_values']
#         model.load_state_dict(state_dict)
#         logging.info(f'Model loaded from {args.load}')

#     model.to(device=device)
#     try:
#         train_model(
#             model=model,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             learning_rate=args.lr,
#             device=device,
#             img_scale=args.scale,
#             val_percent=args.val / 100,
#             amp=args.amp,
#             model_name=args.model
#         )
#     except torch.cuda.OutOfMemoryError:
#         logging.error('Detected OutOfMemoryError! '
#                       'Enabling checkpointing to reduce memory usage, but this slows down training. '
#                       'Consider enabling AMP (--amp) for fast and memory efficient training')
#         torch.cuda.empty_cache()
#         model.use_checkpointing()
#         train_model(
#             model=model,
#             epochs=args.epochs,
#             batch_size=args.batch_size,
#             learning_rate=args.lr,
#             device=device,
#             img_scale=args.scale,
#             val_percent=args.val / 100,
#             amp=args.amp
#         )


