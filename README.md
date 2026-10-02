# Alzheimer’s Disease Detection Using Hippocampus Segmentation with U-Net

## Project Overview

This project proposes a computer-aided research system for analyzing brain Magnetic Resonance Imaging (MRI) scans, segmenting the hippocampus using a U-Net deep-learning model, and extracting hippocampal measurements that may support Alzheimer’s disease research.

The planned application combines an MRI-processing pipeline, hippocampus segmentation, quantitative analysis, optional disease-stage classification, an interactive web interface, and secure record management.

**Important:** This is an academic research prototype, not a clinically validated diagnostic system. Its outputs must not be used as a substitute for professional medical assessment.

## Objectives

- Organize and validate brain MRI data.
- Convert DICOM studies to NIfTI volumes when required.
- Develop a reproducible MRI preprocessing pipeline.
- Prepare and verify hippocampus segmentation masks.
- Train and evaluate a U-Net model for hippocampus segmentation.
- Calculate left and right hippocampal volumes and volume asymmetry.
- Explore optional Alzheimer’s stage classification when suitable labels and data are available.
- Present MRI slices, segmentation overlays, and analysis reports through a web application.
- Provide role-based access for Admin and User accounts.
- Maintain project documentation, quality-control reports, and experiment results.

## Updated System Architecture

The diagram below shows the proposed end-to-end system, including authentication, separate User and Admin workflows, MRI processing, U-Net segmentation, evaluation, feature extraction, visualization, reporting, and database storage.

![Updated System Architecture](docs/system-architecture.png)

### Architecture Workflow

1. **Authentication and role-based access:** The application identifies the logged-in user and applies permissions based on the assigned role.
2. **User dashboard:** A user uploads a brain MRI and views their own scans, results, reports, and history.
3. **Admin dashboard:** An authorized administrator manages user accounts and monitors system activity.
4. **MRI validation:** The backend checks uploaded files and required metadata.
5. **DICOM-to-NIfTI conversion:** DICOM series are converted to NIfTI volumes when the input is in DICOM format.
6. **MRI preprocessing:** The planned pipeline includes resampling, normalization, denoising, bias-field correction, and quality control.
7. **Hippocampus segmentation:** A U-Net model predicts a hippocampus mask from the processed MRI.
8. **Segmentation evaluation:** Where verified ground-truth masks are available, predictions are evaluated using Dice score, Intersection over Union (IoU), precision, and recall.
9. **Hippocampal feature extraction:** The system calculates left and right hippocampal volumes and volume asymmetry from suitable masks.
10. **Optional classification and explainability:** Classification and explainable-AI methods may be explored if appropriate diagnostic labels and sufficient validated data are available.
11. **Results and visualization:** MRI slices and segmentation overlays are displayed in the frontend.
12. **Report generation and storage:** Analysis reports and authorized records are stored in the database.
13. **History and monitoring:** Users can access only their own records; authorized administrators can access permitted management and monitoring functions.

*The architecture represents the planned complete system. Only the components explicitly identified in the project status below should be considered implemented and verified.*

## Project Pipeline

### 1. Dataset Acquisition

The project uses brain MRI data obtained from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), subject to the applicable data-use terms. Dataset access, redistribution, and publication must follow ADNI requirements.

### 2. DICOM-to-NIfTI Conversion

The conversion pipeline uses Python imaging tools to read DICOM series, construct 3D volumes, and save NIfTI files. Conversion reports record successful and failed cases and support quality checks.

### 3. MRI Preprocessing

The intended preprocessing stages include:

- MRI volume validation and orientation checks.
- Resampling to a consistent voxel spacing when appropriate.
- Intensity normalization.
- Denoising.
- Bias-field correction.
- Optional skull stripping or registration when justified by the data and task.
- Quality-control reports and visual inspection.

Preprocessing choices must preserve anatomical information and maintain spatial alignment between MRI volumes and segmentation masks.

### 4. Hippocampus Mask Preparation

Training a supervised segmentation model requires MRI volumes paired with verified ground-truth hippocampus masks. The masks must be checked for correct subject identity, orientation, voxel spacing, dimensions, and alignment. Categorical masks should use nearest-neighbor interpolation if resampling is required.

Training, validation, and test splits should be created at the subject level to prevent data leakage.

### 5. U-Net Segmentation

The planned baseline is a U-Net convolutional neural network. It learns to predict hippocampus labels from MRI input volumes or slices using paired training examples. Possible future experiments include Attention U-Net or 3D U-Net, depending on available data, compute resources, and project scope.

### 6. Evaluation

When verified ground-truth masks are available, evaluation may include Dice similarity coefficient, Intersection over Union (IoU), precision and recall, visual review of segmentation overlays, and error analysis on held-out subjects. Metrics should be calculated on data not used to train the model.

### 7. Hippocampal Analysis

The predicted or reference masks can support quantitative analysis, including left and right hippocampal volumes and asymmetry. Volume calculations require correct voxel-spacing information and validated masks.

### 8. Optional Classification and Explainable AI

Classification is a separate, optional research stage. It depends on suitable clinical or diagnostic labels, an adequate sample size, and a carefully designed subject-level evaluation. Potential methods and explainability techniques will be selected after the data and segmentation pipeline are validated.

## Technology Stack

| Component | Planned / Used Technology |
|---|---|
| Programming language | Python |
| MRI data | ADNI |
| Medical-image conversion and processing | SimpleITK, NiBabel, pydicom |
| Deep learning | PyTorch |
| Segmentation architecture | U-Net |
| Data analysis | NumPy, pandas, scikit-learn |
| Visualization | Matplotlib and web-based MRI visualization |
| Backend API | FastAPI |
| Frontend | React |
| Database | MySQL |
| Development environment | VS Code, Git, GitHub |
| Operating system used for development | Windows |

The technology list describes the project stack and planned implementation; it does not imply that every component has already been integrated.

## User Roles and Access Control

### User

- Register or sign in.
- Upload MRI scans.
- View their own MRI records and analysis results.
- View segmentation overlays and measurements.
- Access their own reports and analysis history.

### Admin

- Manage user accounts and roles.
- Monitor system activity.
- View authorized system records and reports.
- Review audit logs and model-version information.

**Security requirement:** Authentication, role checks, and record ownership must be enforced by the backend. Hiding controls in the frontend alone is not sufficient. Sensitive administrative access should be authorized and logged.

## Proposed Database Design

| Table | Purpose |
|---|---|
| `users` | User profile and authentication-related records |
| `roles` | Role definitions and permissions |
| `mri_scans` | MRI upload metadata and ownership |
| `analysis_results` | Segmentation outputs, measurements, and metrics |
| `reports` | Generated report metadata and storage references |
| `activity_logs` | Security and system activity records |
| `model_versions` | Model identifiers and experiment/version metadata |

The final schema will be refined during backend implementation. Passwords must be stored using secure password hashing, not as plain text.

## Proposed API Endpoints

The following are design examples and should be treated as planned until implemented and tested.

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/auth/register` | Register a user |
| `POST` | `/api/auth/login` | Authenticate a user |
| `GET` | `/api/users/me` | Retrieve the current user profile |
| `POST` | `/api/mri/upload` | Upload an MRI scan |
| `GET` | `/api/mri` | List scans accessible to the current user |
| `POST` | `/api/analysis/{scan_id}` | Start an analysis |
| `GET` | `/api/analysis/{analysis_id}` | Retrieve analysis results |
| `GET` | `/api/reports/{report_id}` | Retrieve an authorized report |
| `GET` | `/api/admin/users` | Admin-only user management |
| `GET` | `/api/admin/activity` | Admin-only activity monitoring |

All endpoints must validate input, enforce access permissions, and return appropriate errors. Actual endpoint names may change during implementation.

## Repository Structure

The repository may evolve as development progresses. A target structure is shown below:

```text
Alzheimer-Hippocampus-Segmentation/
├── README.md
├── docs/
│   └── system-architecture.png
├── preprocessing/
│   ├── dicom_to_nifti.py
│   └── image_preprocessing.py
├── dataset/
│   ├── raw/                 # Keep private; do not commit
│   ├── nifti/               # Keep private; do not commit
│   └── masks/               # Keep private unless permitted
├── models/
│   ├── unet.py
│   └── checkpoints/         # Large model files; do not commit by default
├── backend/
├── frontend/
├── results/
│   ├── plots/
│   └── reports/
├── tests/
├── requirements.txt
└── .gitignore
```

This is a target layout, not a claim that every directory or module currently exists.

## Setup

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies.
- Git.
- VS Code (recommended).
- Access to the required ADNI data.
- Sufficient disk space for MRI volumes and model experiments.

### Clone the Repository

```powershell
git clone https://github.com/minnachirakkal775-bot/Alzheimer-Hippocampus-Segmentation.git
cd Alzheimer-Hippocampus-Segmentation
```

### Create and Activate a Virtual Environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, follow your organization's security policy or use the VS Code interpreter selector to choose the virtual environment.

### Install Dependencies

If a `requirements.txt` file is present and up to date:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install only the dependencies required by the current pipeline. PyTorch installation commands can vary by CPU/GPU and CUDA configuration; use the official PyTorch installation selector for the target machine.

### Run DICOM-to-NIfTI Conversion

After configuring the input and output paths in the conversion script:

```powershell
python preprocessing/dicom_to_nifti.py
```

### Run MRI Preprocessing and Quality Control

After configuring the input directory and output/report locations:

```powershell
python preprocessing/image_preprocessing.py
```

Check the generated reports and plots before using any volumes for model development. Script arguments and paths may differ by local configuration.

## Dataset and Data Handling

- Keep raw ADNI data out of public Git repositories.
- Follow ADNI data-use and citation requirements.
- Do not upload identifiable or restricted medical data to public services.
- Keep subject identifiers and metadata access controlled.
- Preserve MRI-to-mask pairing and spatial metadata.
- Use subject-level splits for model evaluation.
- Document preprocessing parameters and dataset versions for reproducibility.

## Current Project Status

### Verified Progress

- ADNI baseline MRI data has been downloaded and organized locally.
- A conversion pipeline has been run on a five-volume subset.
- Five MRI volumes were converted and validated successfully.
- The converted volumes have dimensions of 256 × 256 × 166, with reported spacing approximately 1.016 × 1.016 × 1.2 mm.
- The five-volume subset includes subjects `005_S_0324`, `005_S_0448`, `005_S_0553`, `005_S_0572`, and `005_S_0602`.
- The conversion report is available at `results/dicom_conversion_report.csv`.
- Initial preprocessing quality-control checks passed for all five volumes, with no warnings or rejections in the reported runs.
- Quality-control reports and slice plots have been generated under `results/`.

### Important Qualification

The five volumes are a small pipeline-debugging subset, not a sufficient basis for claiming model generalization. The current quality-control stage should not be described as complete normalization, resampling, or full MRI preprocessing unless those operations have actually been applied and verified. No completed U-Net training, validated segmentation evaluation, disease classification, full-stack integration, or deployment is claimed here.

### Next Steps

1. Obtain and verify hippocampus ground-truth masks.
2. Confirm MRI-mask alignment and labels.
3. Implement and document the full preprocessing pipeline.
4. Create subject-level train, validation, and test splits.
5. Implement a baseline U-Net and data loader.
6. Train the model and save checkpoints and training history.
7. Evaluate on held-out subjects and review predictions.
8. Calculate hippocampal measurements from validated masks.
9. Explore optional classification only if appropriate labels and sample sizes are available.
10. Develop and test the React frontend and FastAPI backend.
11. Integrate authentication, role-based permissions, and MySQL storage.
12. Perform testing, documentation, and deployment preparation.

## Team Work Distribution

The project is planned for four members with approximately equal overall responsibility. Assignments below describe the planned division, not necessarily completed work.

| Member | Responsibility | Main Deliverables |
|---|---|---|
| Member 1 | Dataset collection and MRI preprocessing | Organized dataset, conversion pipeline, preprocessing, QC reports, and documentation |
| Member 2 | Mask preparation and deep learning | Verified MRI-mask pairs, data splits, U-Net, training pipeline, checkpoints, and predictions |
| Member 3 | Evaluation and hippocampal analysis | Metrics, error analysis, volume measurements, comparisons, plots, and optional classification experiments |
| Member 4 | Frontend, backend, and integration | React interface, FastAPI services, database integration, access control, tests, and deployment preparation |

All members will contribute to code reviews, integration testing, weekly progress updates, final documentation, and presentation preparation.

## Version Control

Use `.gitignore` to exclude virtual environments, raw datasets, generated caches, private configuration, and large artifacts unless there is an explicit reason and permission to version them.

Example commands for committing the README and architecture image:

```powershell
git add README.md docs/system-architecture.png
git commit -m "Update README with system architecture"
git push origin main
```

Review `git status` before committing to ensure that raw MRI data, `.venv`, credentials, and unintended large files are not staged.

## Limitations

- The currently verified MRI subset is small and is intended for pipeline validation.
- Supervised U-Net training requires correctly paired ground-truth masks.
- Segmentation quality depends on data quality, annotation consistency, and independent evaluation.
- Classification performance cannot be established without suitable labels and a defensible validation design.
- The web application and database design are planned components until implementation and testing are complete.
- This research prototype is not approved for clinical diagnosis or treatment decisions.

## Future Enhancements

- Expand the dataset and verify mask availability and quality.
- Compare 2D and 3D segmentation approaches where feasible.
- Evaluate attention-based segmentation architectures.
- Improve preprocessing reproducibility and automated quality checks.
- Add interactive 3D visualization and longitudinal analysis when repeat scans are available.
- Add robust experiment tracking and model versioning.
- Complete secure web integration and deployment.
- Conduct broader external validation before considering any clinical use.

## Acknowledgment

The project uses MRI data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). Any use or publication of ADNI-derived data must follow the initiative’s applicable data-use and acknowledgment requirements.

## Disclaimer

This software is developed for educational and research purposes. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about Alzheimer’s disease.
