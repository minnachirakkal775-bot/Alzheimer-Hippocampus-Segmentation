# Alzheimer’s Disease Detection Using Hippocampus Segmentation with U-Net

An AI-assisted research project that aims to segment the hippocampus from brain MRI scans using a U-Net model and provide analysis through an Android application. The planned mobile app will communicate with a Python backend and may be distributed through the Google Play Store after implementation, testing, and approval.

> **Project status:** Initial dataset conversion and quality checks have been completed for five test MRI volumes. Model training, segmentation evaluation, backend implementation, Android app development, and Play Store publication are still pending. This is a research prototype, not a medical diagnostic device.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Technology Stack](#technology-stack)
- [Dataset and Current Progress](#dataset-and-current-progress)
- [Project Structure](#project-structure)
- [Team Responsibilities](#team-responsibilities)
- [Development Roadmap](#development-roadmap)
- [Setup](#setup)
- [Planned Application Workflow](#planned-application-workflow)
- [Evaluation Plan](#evaluation-plan)
- [Privacy and Security](#privacy-and-security)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Acknowledgment](#acknowledgment)
- [Disclaimer](#disclaimer)

## Project Overview

Alzheimer’s disease is associated with changes in brain structure, including the hippocampus. This project explores the use of deep learning to segment the hippocampus from T1-weighted brain MRI scans. The intended system combines MRI data conversion and preprocessing, U-Net segmentation, evaluation and measurement, a Python backend, and an Android application.

The system is being developed as a research and educational project. It must not be used to make clinical decisions.

## Objectives

- Organize and validate ADNI MRI data.
- Convert DICOM image series to NIfTI volumes.
- Develop a reproducible MRI preprocessing pipeline.
- Obtain and verify hippocampus segmentation masks.
- Train and evaluate a U-Net segmentation model.
- Calculate hippocampal measurements from verified masks.
- Develop backend APIs to connect the model and mobile app.
- Build an Android app for MRI upload, result visualization, and scan history.
- Prepare the app for testing and possible Google Play Store distribution.

## System Architecture

```mermaid
flowchart TD
    A[ADNI MRI Dataset] --> B[DICOM MRI Files]
    B --> C[DICOM to NIfTI Conversion]
    C --> D[MRI Preprocessing and Quality Checks]
    D --> E[Verified Hippocampus Masks]
    E --> F[MRI-Mask Pairing and Subject-Level Splits]
    F --> G[U-Net Training]
    G --> H[Hippocampus Segmentation Prediction]
    H --> I[Evaluation: Dice, IoU, Precision, Recall]
    H --> J[Hippocampal Volume Analysis]
    I --> K[FastAPI Backend]
    J --> K
    L[Flutter Android Application] --> M[Login and MRI Upload]
    M --> K
    K <--> N[MySQL Database]
    K --> O[Segmentation and Measurement Results]
    O --> P[Mobile Result Viewer and Reports]
    P --> Q[Signed Android App Bundle]
    Q --> R[Google Play Console Review and Release]
```

## Technology Stack

| Component | Planned technology | Purpose |
|---|---|---|
| MRI dataset | ADNI | Source of brain MRI data |
| Image format | DICOM, NIfTI | MRI input and processing |
| Development environment | VS Code | Python and backend development |
| Image processing | SimpleITK, NiBabel, NumPy | Reading and processing MRI volumes |
| Data analysis | Pandas, Matplotlib | Reports and visualizations |
| Deep learning | PyTorch | U-Net implementation and training |
| Backend | Python, FastAPI | APIs and model inference |
| Database | MySQL | User, scan, and analysis records |
| Mobile app | Flutter, Dart, Android Studio | Android user interface |
| Version control | Git, GitHub | Collaboration and source management |
| Distribution | Android App Bundle, Google Play Console | App release, subject to requirements and review |

## Dataset and Current Progress

The project uses baseline T1-weighted MRI data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). Follow the dataset provider’s access conditions and data-use agreement.

### Confirmed initial results

| Item | Current result |
|---|---|
| DICOM files identified | 830 |
| MRI volumes converted in the test subset | 5 |
| Successful conversions in that subset | 5 |
| Conversion failures in that subset | 0 |
| Typical converted volume dimensions | 256 × 256 × 166 |
| Approximate voxel spacing | 1.016 × 1.016 × 1.2 mm |
| Initial QC volumes checked | 5 |
| QC warnings/rejections in the reported test run | 0 |

These results describe only the five-volume test subset. The current QC-accepted files should not be described as fully normalized or completely preprocessed; additional preprocessing and validation remain to be done.

### Existing reports and visualizations

- `results/dicom_conversion_report.csv`
- `results/preprocessing_qc_report_*.csv`
- `results/plots/` — MRI slice quality-control plots

Large raw datasets, private data, virtual environments, generated caches, and model weights should not be committed to Git unless the project has an approved storage and sharing plan.

## Project Structure

The repository may evolve as implementation proceeds. A suggested structure is:

```text
Alzheimer-Hippocampus-Segmentation/
├── preprocessing/
│   ├── dicom_to_nifti.py
│   └── image_preprocessing.py
├── backend/
│   ├── app/
│   ├── api/
│   ├── database/
│   └── main.py
├── android_app/
│   ├── lib/
│   ├── android/
│   └── pubspec.yaml
├── models/
├── results/
│   └── plots/
├── documentation/
├── .gitignore
└── README.md
```

Keep original MRI data outside the repository or in an approved private data store. The exact folders may differ from this suggested layout.

## Team Responsibilities

The work is divided into four approximately equal responsibility areas. The percentages refer to planned workload, not a claim that every member has completed that share.

| Member | Responsibility | Main tasks | Deliverables |
|---|---|---|---|
| Member 1 | Dataset collection and MRI preprocessing | Collect and organize ADNI data; convert DICOM to NIfTI; validate dimensions and metadata; perform normalization, resampling, and other suitable preprocessing; prepare and verify masks; create subject-level data splits; generate QC reports. | Organized dataset, conversion/preprocessing scripts, verified MRI-mask pairs, QC reports, split files. |
| Member 2 | U-Net model development and training | Study and implement U-Net; build data loaders; select suitable losses; apply augmentation; train and validate the model; save checkpoints and training history; generate predictions; provide an inference function. | U-Net implementation, training pipeline, trained checkpoint, training graphs, predicted masks, inference interface. |
| Member 3 | Evaluation, hippocampal analysis, and backend | Evaluate segmentation using appropriate metrics; analyze errors; calculate hippocampal volumes and asymmetry; build FastAPI endpoints; connect MySQL; integrate the trained model; implement authentication and access controls; test APIs. | Evaluation report, measurement module, backend APIs, database integration, API documentation. |
| Member 4 | Android app, integration, and release preparation | Develop Flutter screens; implement login, upload, result display, history, and reports; connect APIs; create app icon; test on Android devices; build signed AAB; prepare store listing and testing materials. | Android app, app icon, integrated interface, tested release bundle, Play Store listing materials. |

All members contribute to weekly meetings, project diaries, code reviews, integration testing, final documentation, review presentations, and the final demonstration.

## Development Roadmap

| Phase | Activities | Status |
|---|---|---|
| 1. Research and setup | Topic selection, literature survey, repository, environment, project planning | Initial work completed |
| 2. Dataset acquisition and conversion | ADNI data organization, DICOM-to-NIfTI conversion, initial validation | Completed for five test volumes |
| 3. Full MRI preprocessing | Normalization, resampling, additional suitable processing, QC | Pending |
| 4. Mask preparation | Obtain masks, verify alignment and labels, pair with MRI, subject-level splits | Pending |
| 5. U-Net development | Implement, train, validate, save model, generate predictions | Pending |
| 6. Evaluation and analysis | Dice, IoU, precision, recall, error analysis, volume measurements | Pending |
| 7. Backend | FastAPI, MySQL, authentication, upload and inference endpoints | Pending |
| 8. Android application | Flutter UI, API integration, visualization, history, reports | Pending |
| 9. Testing and release | Integration/device tests, privacy documentation, signed AAB, closed testing, Play Store review | Pending |

## Setup

### Prerequisites

- Windows 10/11 or a compatible development environment.
- Python version compatible with the selected PyTorch and image-processing packages.
- VS Code, Git, Android Studio, and Flutter SDK for mobile development.
- Access to ADNI data under its applicable terms.
- MySQL for the planned backend.

### Clone the repository

```powershell
git clone https://github.com/minnachirakkal775-bot/Alzheimer-Hippocampus-Segmentation.git
cd Alzheimer-Hippocampus-Segmentation
```

### Create and activate a Python virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies from the maintained requirements file when available:

```powershell
pip install -r requirements.txt
```

If a requirements file has not yet been created, install and pin the dependencies used by implemented scripts before sharing the environment. Check package compatibility rather than assuming arbitrary versions will work together.

### Run the existing conversion script

After placing authorized DICOM data in the configured input location and checking the script’s path settings:

```powershell
python preprocessing/dicom_to_nifti.py
```

### Run the existing quality-checking script

```powershell
python preprocessing/image_preprocessing.py
```

Review the generated CSV reports and plots. These scripts currently support the initial conversion/QC workflow; they do not imply that all planned preprocessing steps have been implemented.

### Mobile app setup

After the Flutter project has been created:

```powershell
flutter doctor
flutter pub get
flutter run
```

Configure the API base URL to point to a backend reachable from the emulator or physical device. `localhost` on a phone does not automatically refer to the development computer.

## Planned Application Workflow

1. A user signs in to the Android application.
2. The user selects an MRI file in a supported format.
3. The app sends the file securely to the backend.
4. The backend validates the file and runs the configured preprocessing and model inference pipeline.
5. The model generates a hippocampus segmentation.
6. Measurements are calculated only after the mask and image geometry have been validated.
7. The backend stores authorized scan and analysis records.
8. The app displays the image, segmentation overlay, measurements, and report.
9. Users can view their own scan history. Administrative access must be role-controlled and logged.

## Evaluation Plan

The segmentation model will be evaluated against verified ground-truth masks using held-out subjects. Planned metrics include:

- **Dice similarity coefficient:** overlap between predicted and reference masks.
- **Intersection over Union (IoU):** intersection divided by union of predicted and reference regions.
- **Precision:** proportion of predicted positive voxels that are correct.
- **Recall:** proportion of reference positive voxels recovered by the model.

Report the evaluation split, sample count, metric definitions, and limitations. Do not report performance values until the corresponding experiments have been run and checked.

## Privacy and Security

MRI data can be sensitive. Before making the app available to other users:

- Follow ADNI data-use terms and do not redistribute restricted data.
- Collect only information required by the application.
- Use secure transport for uploads and API requests.
- Enforce authentication, role-based permissions, and record ownership on the backend.
- Do not rely only on hiding controls in the mobile interface for access control.
- Restrict and log administrative access to sensitive records.
- Define secure storage, retention, deletion, backup, and incident-handling procedures.
- Publish an accurate privacy policy and complete applicable Google Play Data safety declarations.
- Use synthetic or otherwise authorized test data during development and demonstrations.

## Limitations

- Initial conversion and QC results cover only five MRI volumes.
- Verified hippocampus masks have not yet been confirmed for the current test subset.
- U-Net training and held-out evaluation have not been completed.
- Segmentation performance and clinical validity are therefore unknown.
- Alzheimer’s stage classification would require suitable labels, adequate data, and separate validation; it is not established by hippocampus segmentation alone.
- The Android app, backend, and Play Store release are planned work, not completed deliverables.

## Future Enhancements

- Complete and validate the full preprocessing pipeline.
- Train and compare baseline U-Net with suitable variants such as Attention U-Net, if resources and data permit.
- Add robust 2D/3D visualization and longitudinal analysis where repeat scans are available.
- Explore classification only with appropriate labels and rigorous validation.
- Add explainability features with clear limitations.
- Improve accessibility, usability, and multilingual support.
- Conduct broader testing and prepare a secure production deployment.

## Acknowledgment

The project uses MRI data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI), subject to its access requirements and data-use conditions. The team acknowledges the researchers, institutions, participants, and supporting organizations involved in making the dataset available.

## Disclaimer

This application is an academic research prototype. It is not intended to diagnose, treat, cure, or prevent Alzheimer’s disease or any other medical condition. Outputs must not replace assessment by qualified healthcare professionals. No claim of clinical accuracy or reliability should be made without appropriate independent validation and regulatory review.
