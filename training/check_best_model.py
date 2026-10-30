
import os
import torch
import numpy as np
from monai.networks.nets import UNet

checkpoint_path = r"results\training_v2\best_model.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(checkpoint_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Re-create the same model architecture used during training.
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128),
    strides=(2, 2, 2),
    num_res_units=2,
).to(device)

# Load the checkpoint safely.
try:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Checkpoint loaded successfully.")
print("Best epoch:", checkpoint.get("epoch", "Unknown"))
print("Validation Dice:", checkpoint.get("val_dice", "Unknown"))
print("Device:", device)

# Verify the model produces valid output.
with torch.no_grad():
    dummy_input = torch.randn(1, 1, 96, 96, 96, device=device)
    logits = model(dummy_input)
    probabilities = torch.softmax(logits, dim=1)
    prediction = probabilities.argmax(dim=1)

print("Output shape:", tuple(logits.shape))
print("Prediction shape:", tuple(prediction.shape))
print("Predicted foreground voxels on test input:",
      int((prediction == 1).sum().item()))
print("Foreground probability range:",
      float(probabilities[:, 1].min()),
      float(probabilities[:, 1].max()))

print("\nCheckpoint verification completed.")
