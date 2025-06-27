import os, zipfile, urllib.request, tqdm

TASK2_FILES = {
    "train_images":  "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip",
    "train_masks":   "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Training_GroundTruth_v3.zip",
    "val_images":    "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Validation_Input.zip",
    "val_masks":     "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Validation_GroundTruth.zip",
    "test_images":   "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip",
    "test_masks":    "https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task2_Test_GroundTruth.zip"
}


root = "isic2018_task2"
os.makedirs(root, exist_ok=True)

def download(url, dest):
    if os.path.exists(dest): return
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:
        total = int(resp.headers["Content-Length"])
        with tqdm.tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                bar.update(len(chunk))

for key, url in TASK2_FILES.items():
    zip_path = os.path.join(root, os.path.basename(url))
    download(url, zip_path)
    subdir   = key.replace("_images", "").replace("_masks", "")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(os.path.join(root, subdir))