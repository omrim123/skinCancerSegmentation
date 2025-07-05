import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
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
    return dice.mean() # Average Dice score across all classes

def dice_coeff_vec(input: Tensor, target: Tensor, epsilon: float = 1e-6):
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
    return dice # Dice score across all classes

def jaccard_index(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    Computes the Jaccard index for multi-label segmentation.
    Expects input of shape (N, C, H, W).
    """
    assert input.size() == target.size()
    assert input.dim() == 4, "Input must be a 4D tensor (N, C, H, W)"

    inter = (input * target).sum(dim=(0, 2, 3))
    union = input.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) - inter
    
    # Handle cases where a class is not present in either input or target
    union = torch.where(union == 0, inter, union)

    jaccard = (inter + epsilon) / (union + epsilon)
    return jaccard.mean() # Average Jaccard score across all classes

def dice_loss(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1 - dice_coeff(input, target)

def dice_loss_vec(input: Tensor, target: Tensor):
    # Dice loss (objective to minimize) between 0 and 1
    return 1.0 - dice_coeff_vec(input, target)

def jaccard_loss(input: Tensor, target: Tensor):
    # Jaccard loss (objective to minimize) between 0 and 1
    return 1 - jaccard_index(input, target)