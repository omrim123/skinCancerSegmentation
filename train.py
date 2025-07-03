# import argparse
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

USE_WANDB = False
if not USE_WANDB:
    os.environ["WANDB_MODE"] = "disabled"
import wandb


from evaluate import evaluate
from models.unet import UNet
from utils.data_loading import ISIC2018Task2
from utils.dice_score import dice_loss

train_dir_img = Path('./isic2018_resized/train/ISIC2018_Task1-2_Training_Input/')
train_dir_mask = Path('./isic2018_resized/train/ISIC2018_Task2_Training_GroundTruth_v3/')

val_dir_img = Path('./isic2018_resized/val/ISIC2018_Task1-2_Validation_Input/')
val_dir_mask = Path('./isic2018_resized/val/ISIC2018_Task2_Validation_GroundTruth/')

dir_checkpoint = Path('./checkpoints/')


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
):
    
    input_size_h = 256
    input_size_w = 256


    train_transformer = PairCompose([
        MyNormalize(mean=157.561, std=26.706),
        ToTensorPair(),
        RandomHorizontalFlipPair(p=0.5),
        RandomVerticalFlipPair(p=0.5),
        # RandomRotationPair(degrees=90),
        # ResizePair(input_size_h, input_size_w),
    ])

    # 1. Create dataset
    try:
        train_set = ISIC2018Task2(train_dir_img, train_dir_mask,transform=train_transformer) # TODO add preprocess
        val_set = ISIC2018Task2(val_dir_img, val_dir_mask,transform=train_transformer)
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
    optimizer = optim.Adam(model.parameters(),
                    lr=learning_rate, weight_decay=weight_decay)    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

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
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        # print("masks_pred shape:", masks_pred.shape)
                        # print("true_masks shape:", true_masks.shape)

                        loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )
                        loss += dice_loss(
                            torch.sigmoid(masks_pred),
                            true_masks,
                            multiclass=False  # For multi-label binary segmentation, not multiclass
                        )

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
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            # state_dict['mask_values'] = train_set.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')



class HyperParams():
    def __init__(self, epochs, batch_size, lr, load, scale, val, amp, bilinear, classes):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.load = load
        self.scale = scale
        self.val = val
        self.amp = amp
        self.bilinear = bilinear
        self.classes = classes


if __name__ == '__main__':
    args = HyperParams(
        epochs=10, 
        batch_size=4,
        lr=0.002,
        load=None,
        # load='checkpoints/checkpoint_epoch1.pth',
        scale=0.5, # delete  
        val=0.2, 
        amp=False, 
        bilinear=True, 
        classes=5
    )

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')    
    
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel 
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )