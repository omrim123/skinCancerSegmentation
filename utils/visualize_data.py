import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Edit this if you use resized data
# MASKS_FOLDER = "../isic2018_resized/train/ISIC2018_Task2_Training_GroundTruth_v3"
MASKS_FOLDER = "../isic2018_resized/val/ISIC2018_Task2_Validation_GroundTruth"

ATTRIBUTES = [
    "globules",
    "milia_like_cyst",
    "negative_network",
    "pigment_network",
    "streaks",
]

counts = {attr: 0 for attr in ATTRIBUTES}

mask_files = os.listdir(MASKS_FOLDER)
for attr in ATTRIBUTES:
    # Find all masks for this attribute
    attr_files = [f for f in mask_files if f"attribute_{attr}" in f]
    for f in attr_files:
        mask_path = os.path.join(MASKS_FOLDER, f)
        mask = np.array(Image.open(mask_path))
        if np.any(mask == 255):
            counts[attr] += 1

# Plot pie chart
labels = list(counts.keys())
sizes = [counts[attr] for attr in labels]
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
plt.title("Distribution of Valid Attribute Masks in ISIC 2018 Validation Set")

# Save the pie chart as an image
plt.savefig("../images/isic2018_attribute_mask_distribution_Validation.png", bbox_inches="tight")
print("Saved pie chart as isic2018_attribute_mask_distribution_validation.png")

plt.show()