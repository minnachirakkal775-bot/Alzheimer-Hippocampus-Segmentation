
import os
import nibabel as nib
import numpy as np
import csv

split_file = r"dataset\final\metadata\train_val_test_split.csv"
image_dir = r"dataset\final\normalized_images"
label_dir = r"dataset\final\labels"

subjects_to_check = ["013_S_0325", "002_S_0295"]

with open(split_file, newline="") as f:
    rows = {
        row["Subject ID"]: row
        for row in csv.DictReader(f)
    }

for subject in subjects_to_check:
    image_path = os.path.join(
        image_dir, f"{subject}_MRI_normalized.nii.gz"
    )
    label_path = os.path.join(
        label_dir, f"{subject}_hippocampus.nii.gz"
    )

    image_nii = nib.load(image_path)
    label_nii = nib.load(label_path)

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32)

    print(f"\nSubject: {subject} ({rows[subject]['Split']})")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Shapes match:", image.shape == label.shape)
    print("Affines match:", np.allclose(
        image_nii.affine, label_nii.affine, atol=1e-4
    ))
    print("Image affine:\n", image_nii.affine)
    print("Label affine:\n", label_nii.affine)
    print("Image intensity min/max:",
          float(np.min(image)), float(np.max(image)))
    print("Label unique values:", np.unique(label)[:20])
    print("Label foreground voxels:", int(np.count_nonzero(label > 0)))
