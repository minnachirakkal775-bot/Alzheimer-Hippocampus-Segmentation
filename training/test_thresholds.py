
import os
import csv
import torch
import numpy as np

from monai.data import CacheDataset, DataLoader
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
checkpoint_path = r"results\training_v2\best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

items = []
with open(split_file, newline="") as f:
    for row in csv.DictReader(f):
        if row["Split"] == "validation":
            subject = row["Subject ID"]
            items.append({
                "subject": subject,
                "image": os.path.join(
                    image_dir, f"{subject}_MRI_normalized.nii.gz"
                ),
                "label": os.path.join(
                    label_dir, f"{subject}_hippocampus.nii.gz"
                ),
            })

for item in items:
    for key in ("image", "label"):
        if not os.path.exists(item[key]):
            raise FileNotFoundError(item[key])

subjects = [x["subject"] for x in items]
data_items = [
    {"image": x["image"], "label": x["label"]} for x in items
]

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

dataset = CacheDataset(
    data=data_items,
    transform=transform,
    cache_rate=0.2,
    num_workers=0,
)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

try:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

scores = {t: [] for t in thresholds}

with torch.no_grad():
    for i, batch in enumerate(loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)
        truth = label[:, 0] > 0

        logits = sliding_window_inference(
            image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )
        foreground_prob = torch.softmax(logits, dim=1)[:, 1]

        print(f"\n{subjects[i]} | Actual voxels: {truth.sum().item()}")

        for threshold in thresholds:
            pred = foreground_prob >= threshold

            intersection = (pred & truth).sum().item()
            pred_count = pred.sum().item()
            true_count = truth.sum().item()
            denominator = pred_count + true_count

            dice = (
                2.0 * intersection / denominator
                if denominator > 0 else 1.0
            )
            scores[threshold].append(dice)

            print(
                f"  Threshold {threshold:.1f}: "
                f"Dice={dice:.4f}, predicted voxels={pred_count}"
            )

print("\nMEAN VALIDATION DICE BY THRESHOLD")
for threshold in thresholds:
    print(
        f"Threshold {threshold:.1f}: "
        f"{np.mean(scores[threshold]):.5f}"
    )

best_threshold = max(thresholds, key=lambda t: np.mean(scores[t]))
print("\nBest tested threshold:", best_threshold)
print("Best mean validation Dice:", np.mean(scores[best_threshold]))
