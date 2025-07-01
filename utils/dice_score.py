import torch
from torch import Tensor



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Computes the Dice coefficient.
    Supports input of shape (N, C, H, W), (C, H, W), or (H, W).
    If input is 4D, flattens N and C into one dimension.
    """
    assert input.size() == target.size()
    # Flatten batch and channel dims if input is 4D
    if input.dim() == 4:
        input = input.flatten(0, 1)
        target = target.flatten(0, 1)
    elif input.dim() == 3:
        pass
    elif input.dim() == 2:
        pass
    else:
        raise ValueError(f"Unsupported input dimensions: {input.shape}")

    inter = 2 * (input * target).sum(dim=(-1, -2))
    sets_sum = input.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()



def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)