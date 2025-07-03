import torch
from torch import Tensor
import torch.nn.functional as F
from tqdm import tqdm

# --- Prerequisite Functions (Your Dice + New Jaccard) ---

def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Your existing function...
    assert input.size() == target.size()
    if input.dim() == 4:
        input = input.flatten(0, 1)
        target = target.flatten(0, 1)
    
    inter = 2 * (input * target).sum(dim=(-1, -2))
    sets_sum = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Your existing function...
    return dice_coeff(input, target, reduce_batch_first, epsilon)

def jaccard_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """Computes the Jaccard coefficient (Intersection over Union)."""
    assert input.size() == target.size()
    if input.dim() == 4:
        input = input.flatten(0, 1)
        target = target.flatten(0, 1)

    intersection = (input * target).sum(dim=(-1, -2))
    union = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - intersection
    jaccard = (intersection + epsilon) / (union + epsilon)
    return jaccard.mean()

# --- MODIFIED EVALUATION FUNCTION ---

def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    jaccard_score = 0  # NEW: Initialize Jaccard score accumulator

    # CRITICAL IMPROVEMENT: Wrap the loop in torch.no_grad()
    # This disables gradient calculations, speeding up evaluation and reducing memory usage.
    with torch.no_grad():
        # iterate over the validation set
        with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                # The target mask type depends on the logic inside the if/else
                mask_true = mask_true.to(device=device)

                # predict the mask
                mask_pred = net(image)

                assert net.n_classes == 5, f"you must have 5 classes, currently: {net.n_classes}"

                # This is your multi-label case
                mask_true = mask_true.float()
                # Convert model output to probabilities
                mask_pred_probs = torch.sigmoid(mask_pred)

                # compute the Dice score for all channels
                dice_score += multiclass_dice_coeff(mask_pred_probs, mask_true, reduce_batch_first=False)
                # NEW: Compute the Jaccard score for all channels
                jaccard_score += jaccard_coeff(mask_pred_probs, mask_true)

    net.train()
    # MODIFIED: Return both averaged scores
    avg_dice = dice_score / max(num_val_batches, 1)
    avg_jaccard = jaccard_score / max(num_val_batches, 1)
    return avg_dice, avg_jaccard