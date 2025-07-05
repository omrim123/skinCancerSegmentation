import os
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class ISIC2018Task2(Dataset):
    # The five attribute suffixes in the ground‐truth folder:
    ATTRIBUTES = [
        "pigment_network",
        "negative_network",
        "streaks",
        "milia_like_cyst",
        "globules",
    ]

    def __init__(self, images_dir, masks_dir, transform=None):
        """
        images_dir: Path to folder with ISIC_XXXXX.jpg
        masks_dir:  Path to folder with ISIC_XXXXX_<attr>.png for each attr
        """
        # TODO: resize to 256x256
        self.images_dir = Path(images_dir)
        self.masks_dir  = Path(masks_dir)
        self.ids        = [p.stem for p in self.images_dir.glob("ISIC_*.jpg")]
        self.transform  = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx): 
        img_id = self.ids[idx]
        # --- load the RGB image ---
        img_path = self.images_dir / f"{img_id}.jpg"
        img = np.array(Image.open(img_path).convert("RGB"))

        # --- load & stack the 5 attribute masks ---
        masks = []
        for attr in self.ATTRIBUTES:
            m_path = self.masks_dir / f"{img_id}_attribute_{attr}.png"
            m     = np.array(Image.open(m_path).convert("L"), dtype=np.uint8)
            # ensure binary 0/1
            m     = (m > 127).astype(np.uint8)
            masks.append(m)
        # # masks: list of (H,W) → stack → (H,W,5)
        # mask = np.stack(masks, axis=-1)

        mask = np.stack(masks, axis=0)  # (5, H, W)
        # print("Final mask shape returned by __getitem__:", mask.shape)

        if self.transform:
            # print(self.transform)
            img, mask = self.transform((img, mask))

        return {'image': img, 'mask': mask}
