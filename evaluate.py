import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.score_loss_functions import *


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    preds = []
    targets = []
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

            preds.append(mask_pred_binary)
            targets.append(mask_true)
    net.train()
    preds = torch.cat(preds, dim=0)
    targets = torch.cat(targets, dim=0)
    dice_score = dice_coeff(preds, targets)
    jaccard_score = jaccard_index(preds, targets)
    return dice_score, jaccard_score