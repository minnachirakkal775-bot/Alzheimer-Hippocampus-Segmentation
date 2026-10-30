import os
import csv
import torch
import numpy as np

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped
)
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
checkpoint_path = r"results\training\unet_epoch_10.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# --------------------------------------------------
# Validation subjects
# --------------------------------------------------

val_items = []

with open(split_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["Split"] == "validation":
            subject = row["Subject ID"]

            val_items.append({
                "subject": subject,
                "image": os.path.join(
                    image_dir,
                    f"{subject}_MRI_normalized.nii.gz"
                ),
                "label": os.path.join(
                    label_dir,
                    f"{subject}_hippocampus.nii.gz"
                )
            })

print("Validation subjects:", len(val_items))

# --------------------------------------------------
# Transforms
# --------------------------------------------------

val_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"])
])

dataset = Dataset(
    data=val_items,
    transform=val_transforms
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2
).to(device)

checkpoint = torch.load(
    checkpoint_path,
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded successfully.")

# --------------------------------------------------
# Dice calculation
# --------------------------------------------------

def dice_score(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    intersection = np.logical_and(pred, target).sum()

    denominator = pred.sum() + target.sum()

    if denominator == 0:
        return 1.0

    return (2.0 * intersection) / denominator

# --------------------------------------------------
# Validation
# --------------------------------------------------

scores = []

with torch.no_grad():

    for batch in loader:

        subject = batch["subject"][0]

        image = batch["image"].to(device)
        label = batch["label"].to(device)

        prediction = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25
        )

        prediction = torch.argmax(
            prediction,
            dim=1,
            keepdim=True
        )

        pred_np = prediction.cpu().numpy()[0, 0]
        label_np = label.cpu().numpy()[0, 0]

        score = dice_score(
            pred_np == 1,
            label_np == 1
        )

        scores.append(score)

        print(
            f"{subject}: Dice = {score:.4f}"
        )

# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n========================================")
print("Validation complete.")
print(f"Subjects evaluated: {len(scores)}")
print(f"Mean Dice: {np.mean(scores):.4f}")
print(f"Median Dice: {np.median(scores):.4f}")
print(f"Minimum Dice: {np.min(scores):.4f}")
print(f"Maximum Dice: {np.max(scores):.4f}")
print("========================================")
