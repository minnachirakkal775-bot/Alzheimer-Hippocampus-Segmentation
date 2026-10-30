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

items = []

with open(split_file, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["Split"] == "validation":
            subject = row["Subject ID"]

            items.append({
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

transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"])
])

dataset = Dataset(
    data=items,
    transform=transforms
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

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

print("Prediction diagnostic")
print("=====================")

with torch.no_grad():

    for batch in loader:

        subject = batch["subject"][0]

        image = batch["image"].to(device)
        label = batch["label"].to(device)

        output = sliding_window_inference(
            inputs=image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25
        )

        probability = torch.softmax(output, dim=1)

        prediction = torch.argmax(
            probability,
            dim=1
        )

        pred_np = prediction.cpu().numpy()[0]
        label_np = label.cpu().numpy()[0]

        predicted_voxels = int(np.sum(pred_np == 1))
        label_voxels = int(np.sum(label_np > 0))

        total_voxels = int(pred_np.size)

        predicted_fraction = predicted_voxels / total_voxels
        label_fraction = label_voxels / total_voxels

        mean_foreground_probability = float(
            probability[0, 1].mean().cpu()
        )

        print(
            f"{subject}: "
            f"prediction_voxels={predicted_voxels}, "
            f"label_voxels={label_voxels}, "
            f"prediction_fraction={predicted_fraction:.6f}, "
            f"label_fraction={label_fraction:.6f}, "
            f"mean_fg_probability={mean_foreground_probability:.6f}"
        )

print("\nDiagnostic complete.")
