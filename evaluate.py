import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import *


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    jaccard_score = 0
    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)

            # predict the mask
            mask_pred = net(image)

          
            mask_true = mask_true.float()
            # Apply sigmoid and threshold to get binary prediction masks
            mask_pred_probs = torch.sigmoid(mask_pred)
            mask_pred_binary = (mask_pred_probs > 0.5).float()

            # compute the Dice score for all channels using the corrected function
            dice_score += dice_coeff(mask_pred_binary, mask_true)
            jaccard_score += jaccard_index(mask_pred_binary, mask_true)
    net.train()
    return dice_score / max(num_val_batches, 1), jaccard_score / max(num_val_batches, 1)