
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

# Load validation subjects
val_items = []

with open(split_file, newline="") as f:
    for row in csv.DictReader(f):
        if row["Split"] == "validation":
            subject = row["Subject ID"]
            image_path = os.path.join(
                image_dir, f"{subject}_MRI_normalized.nii.gz"
            )
            label_path = os.path.join(
                label_dir, f"{subject}_hippocampus.nii.gz"
            )

            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)
            if not os.path.exists(label_path):
                raise FileNotFoundError(label_path)

            val_items.append({
                "subject": subject,
                "image": image_path,
                "label": label_path,
            })

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

# Keep subject IDs separate from the dictionaries passed to MONAI
subjects = [item["subject"] for item in val_items]
data_items = [
    {"image": item["image"], "label": item["label"]}
    for item in val_items
]

dataset = CacheDataset(
    data=data_items,
    transform=transform,
    cache_rate=0.2,
    num_workers=0,
)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

# Rebuild and load the trained model
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

scores = []

print("Checkpoint epoch:", checkpoint["epoch"])
print("Stored validation Dice:", checkpoint["val_dice"])
print("\nREAL VALIDATION RESULTS")
print("-" * 76)

with torch.no_grad():
    for index, batch in enumerate(loader):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        logits = sliding_window_inference(
            image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )

        # Foreground class = 1
        prediction = logits.argmax(dim=1) == 1
        truth = label[:, 0] > 0

        intersection = (prediction & truth).sum().item()
        predicted_count = prediction.sum().item()
        actual_count = truth.sum().item()

        denominator = predicted_count + actual_count
        dice = (
            2.0 * intersection / denominator
            if denominator > 0 else 1.0
        )

        scores.append(dice)

        print(
            f"{subjects[index]} | "
            f"Dice: {dice:.4f} | "
            f"Predicted: {predicted_count} | "
            f"Actual: {actual_count}"
        )

print("-" * 76)
print(f"Mean validation Dice: {np.mean(scores):.5f}")
print(f"Median validation Dice: {np.median(scores):.5f}")
print(f"Subjects evaluated: {len(scores)}")
