import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

# 1) A Compose that threads an (img,mask) tuple through your transforms
class PairCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        img, mask = data
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


# 2) A ToTensor that handles both image and mask
# Corrected ToTensorPair class in utils.py
class ToTensorPair:
    def __call__(self, img, mask):
        # img: H×W×C numpy array
        # mask: 5×H×W numpy array
        
        # Convert image to tensor C×H×W in [0,1]
        img_t = TF.to_tensor(img) 
        
        # Convert mask numpy array to a float tensor
        mask_t = torch.from_numpy(mask).float()
        
        return img_t, mask_t

# 3) Random flips & rotation for both, using functional API
class RandomHorizontalFlipPair:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.hflip(img), TF.hflip(mask)
        return img, mask

class RandomVerticalFlipPair:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, mask):
        if random.random() < self.p:
            return TF.vflip(img), TF.vflip(mask)
        return img, mask

class RandomRotationPair:
    def __init__(self, degrees):
        self.degrees = degrees
    def __call__(self, img, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        # image: use bilinear (default); mask: nearest
        return (
            TF.rotate(img, angle, interpolation=Image.BILINEAR),
            TF.rotate(mask, angle, interpolation=Image.NEAREST),
        )


# 4) A Resize that also keeps mask integer
class ResizePair:
    def __init__(self, h, w):
        self.size = (h, w)
    def __call__(self, img, mask):
        img_r  = TF.resize(img, self.size, interpolation=Image.BICUBIC)
        mask_r = TF.resize(mask, self.size, interpolation=Image.NEAREST)
        return img_r, mask_r


# 5)
class MyNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, img, mask):
        
        normalized_img = TF.normalize(img, self.mean, self.std)
        
        return normalized_img, mask