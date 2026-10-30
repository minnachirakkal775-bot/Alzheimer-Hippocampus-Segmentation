
import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference

subject = "013_S_0325"

image_path = rf"dataset\final\normalized_images\{subject}_MRI_normalized.nii.gz"
label_path = rf"dataset\final\labels\{subject}_hippocampus.nii.gz"
checkpoint_path = r"results\training_v2\best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

image_nii = nib.load(image_path)
label_nii = nib.load(label_path)

image = image_nii.get_fdata(dtype=np.float32)
label = label_nii.get_fdata(dtype=np.float32) > 0

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

input_tensor = torch.from_numpy(image.copy()).float()
input_tensor = input_tensor.unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    logits = sliding_window_inference(
        input_tensor,
        roi_size=(96, 96, 96),
        sw_batch_size=1,
        predictor=model,
        overlap=0.25,
    )
    foreground_prob = torch.softmax(logits, dim=1)[:, 1]
prediction = (foreground_prob >= 0.3)[0].cpu().numpy()

# Find the axial slice with the largest ground-truth foreground area.
slice_areas = label.sum(axis=(0, 1))
slice_index = int(np.argmax(slice_areas))

mri_slice = image[:, :, slice_index]
true_slice = label[:, :, slice_index]
pred_slice = prediction[:, :, slice_index]

# Robust display scaling for MRI intensities.
vmin, vmax = np.percentile(image, [1, 99])

fig, axes = plt.subplots(1, 4, figsize=(16, 5))

axes[0].imshow(mri_slice.T, cmap="gray", origin="lower",
               vmin=vmin, vmax=vmax)
axes[0].set_title("MRI")

axes[1].imshow(mri_slice.T, cmap="gray", origin="lower",
               vmin=vmin, vmax=vmax)
axes[1].imshow(true_slice.T, cmap="Greens", alpha=0.6,
               origin="lower")
axes[1].set_title("Ground-truth label")

axes[2].imshow(mri_slice.T, cmap="gray", origin="lower",
               vmin=vmin, vmax=vmax)
axes[2].imshow(pred_slice.T, cmap="Reds", alpha=0.6,
               origin="lower")
axes[2].set_title("Predicted mask")

axes[3].imshow(mri_slice.T, cmap="gray", origin="lower",
               vmin=vmin, vmax=vmax)
axes[3].imshow(true_slice.T, cmap="Greens", alpha=0.55,
               origin="lower")
axes[3].imshow(pred_slice.T, cmap="Reds", alpha=0.45,
               origin="lower")
axes[3].set_title("Overlay: green=true, red=prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
output_path = r"results\training_v2\prediction_visualization.png"
plt.savefig(output_path, dpi=160, bbox_inches="tight")
plt.show()

print("Selected slice:", slice_index)
print("Visualization saved to:", output_path)
print("Ground-truth voxels on slice:", int(true_slice.sum()))
print("Predicted voxels on slice:", int(pred_slice.sum()))
