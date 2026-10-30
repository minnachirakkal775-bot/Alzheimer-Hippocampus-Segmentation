
import csv
import os
import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"
checkpoint_path = r"results\training_v2\best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose one training subject and one validation subject.
wanted = {"013_S_0325", "002_S_0295"}
subjects = {}

with open(split_file, newline="") as f:
    for row in csv.DictReader(f):
        subject = row["Subject ID"]
        if subject in wanted:
            subjects[subject] = row["Split"]

if wanted - subjects.keys():
    raise RuntimeError(f"Subjects missing from split CSV: {wanted - subjects.keys()}")

transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"]),
])

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

print("Checkpoint epoch:", checkpoint["epoch"])
print("Comparing training and validation subjects\n")

for subject, split in subjects.items():
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

    data = transform(item)
    image = data["image"].unsqueeze(0).to(device)
    label = data["label"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=(96, 96, 96),
            sw_batch_size=1,
            predictor=model,
            overlap=0.25,
        )
        prediction = logits.argmax(dim=1) == 1

    truth = label[:, 0] > 0
    intersection = (prediction & truth).sum().item()
    predicted_count = prediction.sum().item()
    actual_count = truth.sum().item()
    dice = (
        2 * intersection / (predicted_count + actual_count)
        if predicted_count + actual_count > 0 else 1.0
    )

    print(f"Subject: {subject} ({split})")
    print(f"  Image shape: {tuple(image.shape)}")
    print(f"  Label foreground voxels: {actual_count}")
    print(f"  Predicted foreground voxels: {predicted_count}")
    print(f"  Intersection voxels: {intersection}")
    print(f"  Dice: {dice:.5f}\n")
