# Alzheimer’s Disease Detection Using Hippocampus Segmentation with U-Net

## 📌 Project Overview

Alzheimer’s disease is a progressive neurological disorder that affects memory, thinking, and behavior. Early detection and accurate analysis of brain structures can support the study of disease progression.

This project focuses on hippocampus segmentation from brain Magnetic Resonance Imaging (MRI) scans using a deep learning-based U-Net architecture. The project uses T1-weighted MRI images from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) dataset.

The primary objective is to develop a deep learning model that can identify and segment the hippocampus region from MRI images and evaluate the segmentation performance.

## 🎯 Objectives

* Study and analyze the ADNI MRI dataset.
* Convert DICOM images into NIfTI format.
* Perform MRI preprocessing and intensity normalization.
* Prepare hippocampus segmentation masks.
* Develop a U-Net-based segmentation model.
* Train and validate the model using prepared MRI data.
* Evaluate segmentation performance using appropriate metrics.
* Visualize the predicted hippocampus regions.

## 🧠 Technologies Used

| Technology     | Purpose                               |
| -------------- | ------------------------------------- |
| Python         | Programming language                  |
| PyTorch        | Deep learning framework               |
| U-Net          | Hippocampus segmentation architecture |
| SimpleITK      | DICOM image processing                |
| NiBabel        | NIfTI image handling                  |
| NumPy          | Numerical operations                  |
| OpenCV         | Image processing                      |
| Matplotlib     | Visualization                         |
| Scikit-learn   | Dataset splitting and evaluation      |
| VS Code        | Development environment               |
| Git and GitHub | Version control                       |

## 📂 Project Structure

```text
Alzheimer-Hippocampus-Segmentation/
│
├── README.md
├── .gitignore
├── requirements.txt
│
├── dataset/
│   ├── raw/
│   ├── nifti/
│   ├── preprocessed/
│   ├── masks/
│   └── splits/
│
├── preprocessing/
│   ├── dicom_to_nifti.py
│   ├── image_preprocessing.py
│   ├── normalization.py
│   ├── resampling.py
│   ├── mask_preparation.py
│   └── dataset_split.py
│
├── models/
│   ├── unet.py
│   └── model_config.py
│
├── training/
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
│
├── results/
│   ├── plots/
│   ├── predictions/
│   ├── segmentation/
│   └── checkpoints/
│
├── documentation/
│   ├── preprocessing_report.md
│   ├── dataset_statistics.md
│   └── project_notes.md
│
├── app/
│   └── app.py
│
└── main.py
```

## 🔄 Project Workflow

1. **Dataset Collection:** Obtain T1-weighted MRI scans from the ADNI dataset.
2. **DICOM to NIfTI Conversion:** Convert the downloaded DICOM images into NIfTI volumes.
3. **Preprocessing:** Perform quality checks, orientation handling, and image preparation.
4. **Normalization:** Normalize MRI intensity values.
5. **Resampling:** Prepare images with consistent dimensions and spacing.
6. **Mask Preparation:** Obtain and verify corresponding hippocampus segmentation masks.
7. **Dataset Splitting:** Divide the data into training, validation, and testing sets at the subject level.
8. **Model Development:** Implement the U-Net architecture.
9. **Model Training:** Train the network using MRI images and their corresponding masks.
10. **Evaluation:** Measure segmentation performance using Dice coefficient, IoU, precision, and recall.
11. **Visualization:** Display the original MRI images, ground-truth masks, and predicted segmentation results.

## 📊 Evaluation Metrics

The model will be evaluated using the following metrics:

* **Dice Similarity Coefficient (DSC):** Measures the overlap between predicted and ground-truth segmentation masks.
* **Intersection over Union (IoU):** Measures the ratio of intersection to union between predicted and actual regions.
* **Precision:** Measures the proportion of predicted hippocampus pixels that are correct.
* **Recall:** Measures the proportion of actual hippocampus pixels correctly identified.

## 📁 Dataset

This project uses T1-weighted MRI data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI).

**Dataset source:** [ADNI Official Website](https://adni.loni.usc.edu/)

Access to ADNI data is subject to its data-use policies and approval requirements.

Note: The raw MRI dataset and any restricted patient data are not included in this GitHub repository. Corresponding hippocampus ground-truth segmentation masks must be obtained or prepared separately for supervised model training.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Alzheimer-Hippocampus-Segmentation.git
```

### 2. Navigate to the project directory

```bash
cd Alzheimer-Hippocampus-Segmentation
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the environment

**Windows:**

```bash
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Execution

The project scripts will be executed in the following order as they are implemented:

```bash
python preprocessing/dicom_to_nifti.py
python preprocessing/image_preprocessing.py
python preprocessing/normalization.py
python preprocessing/dataset_split.py
python training/train.py
python training/evaluate.py
```

The scripts may require configuration of dataset paths and annotation locations before execution.

## 👥 Project Team

**Project Title:** Alzheimer’s Disease Detection Using Hippocampus Segmentation with U-Net

**Project Type:** Deep Learning / Medical Image Segmentation

**Development Environment:** Python, VS Code, GitHub

## 📌 Project Status

🚧 **In Development**

The project is being developed in stages, beginning with dataset organization and MRI preprocessing, followed by U-Net implementation, training, and evaluation.

## 📜 License

This project is intended for academic and research purposes. Dataset usage is subject to the terms and conditions of ADNI.

---

**Keywords:** Alzheimer’s Disease, Hippocampus Segmentation, Brain MRI, ADNI, Deep Learning, U-Net, Medical Image Processing, Image Segmentation.
