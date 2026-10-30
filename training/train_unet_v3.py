
import os
import csv
import torch

from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandCropByPosNegLabeld,
)
from monai.networks.nets import UNet
from monai.losses import DiceFocalLoss
from monai.inferers import sliding_window_inference

# --------------------------------------------------
# Paths: preserve all previous experiments
# --------------------------------------------------
split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
checkpoint_dir = r"results\training_v3"

os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# Load training and validation subjects
# --------------------------------------------------
train_items, val_items = [], []

with open(split_file, newline="") as f:
    for row in csv.DictReader(f):
        subject = row["Subject ID"]

        item = {
            "image": os.path.join(
                image_dir, f"{subject}_MRI_normalized.nii.gz"
            ),
            "label": os.path.join(
                label_dir, f"{subject}_hippocampus.nii.gz"
            ),
        }

        for path in item.values():
            if not os.path.exists(path):
                raise FileNotFoundError(path)

        if row["Split"] == "train":
            train_items.append(item)
        elif row["Split"] == "validation":
            val_items.append(item)

if not train_items or not val_items:
    raise RuntimeError("Training or validation split is empty.")

print("Training subjects:", len(train_items))
print("Validation subjects:", len(val_items))

# --------------------------------------------------
# Transforms: retain the existing crop strategy
# --------------------------------------------------
train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96, 96, 96),
        pos=4,
        neg=1,
        num_samples=2,
        image_key="image",
        image_threshold=0,
    ),
])

val_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

train_ds = CacheDataset(
    data=train_items,
    transform=train_transform,
    cache_rate=0.2,
    num_workers=0,
)

val_ds = CacheDataset(
    data=val_items,
    transform=val_transform,
    cache_rate=0.2,
    num_workers=0,
)

train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=0
)

# --------------------------------------------------
# Model: same architecture for a fair comparison
# --------------------------------------------------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

# Experiment: Dice + Focal Loss
loss_function = DiceFocalLoss(
    to_onehot_y=True,
    softmax=True,
    gamma=2.0,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Fixed validation threshold for this experiment
validation_threshold = 0.3

# --------------------------------------------------
# Validation: mean per-subject full-volume Dice
# --------------------------------------------------
def validate():
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)

            logits = sliding_window_inference(
                image,
                roi_size=(96, 96, 96),
                sw_batch_size=1,
                predictor=model,
                overlap=0.25,
            )

            foreground_prob = torch.softmax(logits, dim=1)[:, 1]
            pred = foreground_prob >= validation_threshold
            truth = label[:, 0] > 0

            intersection = (pred & truth).sum(
                dim=(1, 2, 3)
            ).float()

            pred_count = pred.sum(dim=(1, 2, 3)).float()
            true_count = truth.sum(dim=(1, 2, 3)).float()

            dice = (
                2.0 * intersection
                / (pred_count + true_count).clamp_min(1.0)
            )

            dice_scores.extend(dice.cpu().tolist())

    return sum(dice_scores) / len(dice_scores)

# --------------------------------------------------
# Training
# --------------------------------------------------
epochs = 50
best_dice = -1.0

print("\nStarting V3 experiment")
print("Loss: DiceFocalLoss")
print("Validation threshold:", validation_threshold)
print("Maximum epochs:", epochs)

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    steps = 0

    for batch in train_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = loss_function(outputs, labels)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1

    average_loss = running_loss / max(steps, 1)
    val_dice = validate()

    print(
        f"Epoch {epoch:02d}/{epochs} | "
        f"Train loss: {average_loss:.4f} | "
        f"Validation Dice@0.3: {val_dice:.5f}"
    )

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": average_loss,
        "val_dice": val_dice,
        "validation_threshold": validation_threshold,
        "loss_name": "DiceFocalLoss",
    }

    torch.save(
        checkpoint,
        os.path.join(checkpoint_dir, "latest.pth"),
    )

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(
            checkpoint,
            os.path.join(checkpoint_dir, "best_model.pth"),
        )
        print("  New best checkpoint saved.")

print("\nV3 training completed.")
print("Best validation Dice@0.3:", best_dice)
print("Checkpoints saved in:", checkpoint_dir)
