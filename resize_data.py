import os
from PIL import Image
import shutil

def resize_folder(input_folder, output_folder, size=(256, 256), is_mask=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            in_path = os.path.join(input_folder, filename)
            out_path = os.path.join(output_folder, filename)
            img = Image.open(in_path)
            interp = Image.NEAREST if is_mask else Image.BICUBIC
            img_resized = img.resize(size, interp)
            img_resized.save(out_path)
            print(f"Resized {in_path} -> {out_path}")

DATA_FOLDERS = [
    ("isic2018_task2/train/ISIC2018_Task1-2_Training_Input", "isic2018_resized/train/ISIC2018_Task1-2_Training_Input"),
    ("isic2018_task2/train/ISIC2018_Task2_Training_GroundTruth_v3", "isic2018_resized/train/ISIC2018_Task2_Training_GroundTruth_v3"),
    ("isic2018_task2/val/ISIC2018_Task1-2_Validation_Input", "isic2018_resized/val/ISIC2018_Task1-2_Validation_Input"),
    ("isic2018_task2/val/ISIC2018_Task2_Validation_GroundTruth", "isic2018_resized/val/ISIC2018_Task2_Validation_GroundTruth"),
    ("isic2018_task2/test/ISIC2018_Task1-2_Test_Input", "isic2018_resized/test/ISIC2018_Task1-2_Test_Input"),
    ("isic2018_task2/test/ISIC2018_Task2_Test_GroundTruth", "isic2018_resized/test/ISIC2018_Task2_Test_GroundTruth"),
]

for input_folder, output_folder in DATA_FOLDERS:
    if "mask" in input_folder.lower() or "groundtruth" in input_folder.lower():
        resize_folder(input_folder, output_folder, size=(256, 256), is_mask=True)
    else:
        resize_folder(input_folder, output_folder, size=(256, 256), is_mask=False)