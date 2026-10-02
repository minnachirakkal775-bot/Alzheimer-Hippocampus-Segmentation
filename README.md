# Alzheimer's Disease Detection Using Hippocampus Segmentation with U-Net

An academic research project exploring hippocampal segmentation from
brain MRI using deep learning, with planned feature extraction,
classification experiments, an interactive web application, and
deployment.

> **Project status:** Initial dataset conversion and quality-control
> stages are complete for five MRI volumes. Ground-truth mask
> preparation, model training, evaluation, classification, application
> development, and deployment are planned/in progress and must not be
> considered completed until implemented and tested.
>
> **Research-use notice:** This project is for educational and research
> purposes. Its outputs are not a clinical diagnosis or a substitute for
> professional medical assessment.

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Objectives](#objectives)
-   [Project Status](#project-status)
-   [System Architecture](#system-architecture)
-   [End-to-End Pipeline](#end-to-end-pipeline)
-   [Tools and Technologies](#tools-and-technologies)
-   [Repository Structure](#repository-structure)
-   [Dataset](#dataset)
-   [Completed Work](#completed-work)
-   [Planned Modules](#planned-modules)
-   [Installation](#installation)
-   [How to Run](#how-to-run)
-   [Expected Outputs](#expected-outputs)
-   [Evaluation Plan](#evaluation-plan)
-   [Team Responsibilities](#team-responsibilities)
-   [Reproducibility and Data Safety](#reproducibility-and-data-safety)
-   [Limitations](#limitations)
-   [Future Enhancements](#future-enhancements)
-   [Acknowledgements](#acknowledgements)

------------------------------------------------------------------------

## Overview

Alzheimer's disease is a progressive neurological condition. This
project investigates whether deep-learning-based hippocampus
segmentation from structural brain MRI can support quantitative research
and downstream analysis.

The core technical component is a U-Net-based segmentation model. The
broader planned system includes MRI preprocessing, hippocampal volume
measurements, model evaluation, optional classification experiments when
suitable labels are available, interactive visualization, report
generation, and a web application.

## Objectives

1.  Organize and prepare structural MRI data.
2.  Convert DICOM series into NIfTI volumes while preserving spatial
    metadata.
3.  perform quality checks and visualize MRI volumes.
4.  Obtain and validate ground-truth hippocampus masks.
5.  Train and evaluate a U-Net segmentation model.
6.  Explore additional segmentation architectures under a consistent
    evaluation protocol.
7.  Extract quantitative measurements from validated masks.
8.  Conduct classification experiments only when appropriate
    subject-level labels are available.
9.  Build a user interface and backend for research demonstrations.
10. Document limitations, reproducibility, and results.

## Project Status

  -----------------------------------------------------------------------
  Stage                               Status
  ----------------------------------- -----------------------------------
  Literature survey and planning      Completed/ongoing documentation

  ADNI dataset acquisition            Completed for the currently
                                      downloaded subset

  DICOM-to-NIfTI conversion           Completed for 5 series; 0 reported
                                      failures

  NIfTI validation                    Passed for all 5 converted volumes

  MRI preprocessing and QC            Completed for 5 volumes; all passed
                                      implemented checks

  Ground-truth mask preparation       Pending

  Subject-level data splitting        Pending

  U-Net implementation and training   Pending

  Segmentation evaluation             Pending

  Feature extraction and              Pending
  classification                      

  Frontend and backend                Planned

  Integration and testing             Pending

  Deployment                          Planned

  Final report and presentation       Ongoing
  -----------------------------------------------------------------------

Progress percentages are estimates and should be updated from verified
deliverables rather than treated as measured performance.

## System Architecture

``` mermaid
flowchart TD
    A[ADNI Structural MRI] --> B[DICOM Discovery and Series Grouping]
    B --> C[Spatial Slice Ordering]
    C --> D[SimpleITK Volume Reconstruction]
    D --> E[NIfTI Export]
    E --> F[NiBabel Validation]
    F --> G[MRI Quality Control and Visualization]
    G --> H[Preprocessing and Dataset Preparation]
    H --> I[Ground-Truth Hippocampus Masks]
    I --> J[Image-Mask Alignment Checks]
    J --> K[Subject-Level Train / Validation / Test Split]
    K --> L[U-Net Training]
    L --> M[Segmentation Predictions]
    M --> N[Segmentation Evaluation]
    N --> O[Hippocampal Feature and Volume Extraction]
    O --> P[Optional Classification Experiments]
    M --> Q[2D / 3D Visualization]
    P --> R[Backend API]
    Q --> R
    R --> S[Frontend Dashboard]
    S --> T[Reports, History, and Export]
    T --> U[Integration, Testing, and Deployment]
```

## End-to-End Pipeline

### Stage 1 --- Dataset acquisition

-   Download permitted structural MRI data from ADNI.
-   Organize scans by subject and acquisition series.
-   Maintain subject identifiers and relevant metadata.
-   Follow applicable dataset access and usage terms.

### Stage 2 --- DICOM to NIfTI conversion

-   Discover DICOM files recursively.
-   Group slices using `SeriesInstanceUID`.
-   Order slices using spatial metadata where available.
-   Reconstruct volumes using SimpleITK.
-   Save compressed NIfTI files (`.nii.gz`).
-   Validate output dimensions and voxel data with NiBabel.
-   Record conversion outcomes in a CSV report.

### Stage 3 --- MRI quality control and preprocessing

-   Discover NIfTI volumes.
-   Inspect dimensionality, spacing, and signal validity.
-   Check for non-finite values, blank volumes, and other configured QC
    conditions.
-   Generate axial, coronal, and sagittal visualizations.
-   Save QC results to CSV.
-   Apply normalization or resampling only through explicitly
    implemented and documented transforms. The current QC script copies
    validated volumes unchanged.

### Stage 4 --- Ground-truth mask preparation

-   Obtain valid hippocampus segmentation labels compatible with the
    selected scans.
-   Verify image-mask correspondence and spatial alignment.
-   Use nearest-neighbor interpolation for label masks if resampling is
    required.
-   Review overlays before training.
-   Do not create fabricated labels or treat model predictions as ground
    truth.

### Stage 5 --- Dataset splitting

-   Split data by **subject**, not by individual slices, to reduce data
    leakage.
-   Keep training, validation, and test sets separate.
-   Document random seeds and split assignments.

### Stage 6 --- Model development

-   Establish a baseline U-Net.
-   Train with paired MRI volumes and reference masks.
-   Track training and validation loss.
-   Save model checkpoints and configuration.
-   Consider Attention U-Net or 3D U-Net as optional experiments if
    compute resources and dataset size permit.

### Stage 7 --- Evaluation

-   Evaluate predictions on held-out subjects.
-   Report Dice score, IoU, precision, recall/sensitivity, and other
    relevant metrics.
-   Include qualitative overlays and error analysis.
-   Avoid reporting performance until experiments have actually been
    run.

### Stage 8 --- Feature extraction and optional classification

-   Calculate hippocampal volume from valid masks and voxel spacing.
-   Explore classification only when suitable subject-level diagnostic
    labels are available.
-   Prevent leakage during feature selection, model tuning, and
    evaluation.
-   Treat classification as a research experiment, not a clinical
    diagnostic tool.

### Stage 9 --- Application and deployment

Planned components: - Frontend for scan upload and result viewing. -
Backend API for validation and model inference. - 2D slice viewer and
optional 3D visualization. - Analysis history and downloadable
reports. - Testing, containerization, and deployment. -
Privacy-conscious handling of medical imaging data.

------------------------------------------------------------------------

## Tools and Technologies

  -----------------------------------------------------------------------
  Area                    Tool / Technology       Purpose
  ----------------------- ----------------------- -----------------------
  Development environment Visual Studio Code      Code editing and
                                                  project execution

  Language                Python                  Data processing and
                                                  model development

  DICOM handling          pydicom                 Reading DICOM headers
                                                  and metadata

  Medical image I/O       SimpleITK               DICOM series
                                                  reconstruction and
                                                  NIfTI writing

  NIfTI validation        NiBabel                 Reading and validating
                                                  NIfTI volumes

  Numerical processing    NumPy                   Array operations

  Data reports            pandas                  CSV report generation
                                                  and tabular analysis

  Visualization           Matplotlib              MRI slice and result
                                                  plots

  Progress display        tqdm (if used)          Progress reporting

  Deep learning           PyTorch or TensorFlow   U-Net implementation
                          (to be selected)        and training

  Medical imaging         MONAI (optional)        Medical image
  framework                                       transforms and model
                                                  utilities

  Backend                 FastAPI (planned)       Inference and
                                                  application API

  Frontend                React or Streamlit (to  User interface
                          be selected)            

  Database                SQLite initially /      Metadata and analysis
                          PostgreSQL if needed    history

  Packaging               Docker (planned)        Reproducible deployment

  Version control         Git and GitHub          Source-code management
  -----------------------------------------------------------------------

The exact framework versions should be recorded in a dependency file
once the environment is finalized.

## Repository Structure

``` text
Alzheimer-Hippocampus-Segmentation/
├── dataset/
│   ├── nifti/                 # Converted MRI volumes
│   ├── preprocessed/          # QC-passed copies / processed data
│   ├── masks/                 # Ground-truth masks (to be prepared)
│   └── splits/                # Subject-level split files (planned)
├── preprocessing/
│   ├── dicom_to_nifti.py      # DICOM conversion
│   └── image_preprocessing.py  # MRI QC and visualization
├── models/                    # Model definitions (planned)
├── training/                  # Training scripts (planned)
├── evaluation/                # Metrics and analysis (planned)
├── backend/                   # API service (planned)
├── frontend/                  # User interface (planned)
├── results/
│   ├── plots/                 # QC visualizations
│   ├── dicom_conversion_report.csv
│   └── preprocessing_qc_report_*.csv
├── docs/                      # Architecture, reports, and documentation
├── requirements.txt           # To be maintained
└── README.md
```

Actual filenames may differ as the repository evolves. Do not remove
original ADNI data to match this illustrative structure.

## Dataset

The project currently uses a subset of ADNI structural MRI data.

-   Current conversion result: 5 MRI volumes.
-   Each reported volume: 256 × 256 × 166 voxels.
-   Approximate voxel spacing: 1.016 × 1.016 × 1.2 mm.
-   Conversion report: `results/dicom_conversion_report.csv`.
-   QC report: `results/preprocessing_qc_report_20261002_072759.csv`
    (the timestamp changes on subsequent runs).
-   QC outcome: 5 valid volumes, 0 warnings, and 0 rejected volumes in
    the reported run.

The current five-volume subset is suitable for pipeline development and
debugging, but is not by itself evidence of robust model generalization.
A larger, properly labeled dataset may be needed for training and
evaluation.

## Completed Work

### DICOM conversion

The conversion script performs header-based discovery, groups files by
series, reconstructs volumes with SimpleITK, writes compressed NIfTI
files, validates them with NiBabel, and records results in a CSV report.

### MRI QC

The preprocessing script found five NIfTI files. All five passed the
configured quality checks. The script copied the files into
`dataset/preprocessed/` without changing voxel intensities and generated
slice visualizations and a timestamped QC report.

## Planned Modules

-   `dataset_split.py`: subject-level splitting.
-   `mask_preparation.py`: mask discovery, alignment, and validation.
-   `normalization.py`: documented intensity normalization.
-   `resampling.py`: spatial resampling with appropriate interpolation
    for images and masks.
-   `models/unet.py`: baseline U-Net.
-   `training/train.py`: model training.
-   `evaluation/`: segmentation metrics and plots.
-   `backend/` and `frontend/`: application implementation.

These are planned module responsibilities; implementation status should
be checked in the repository.

## Installation

### 1. Clone the repository

``` powershell
git clone https://github.com/minnachirakkal775-bot/Alzheimer-Hippocampus-Segmentation.git
cd Alzheimer-Hippocampus-Segmentation
```

If your local project has an additional nested directory, open the
folder containing `preprocessing/` and `.venv/` as the project root.

### 2. Create and activate a virtual environment (if needed)

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

The currently used conversion and QC scripts require packages such as:

``` powershell
python -m pip install pydicom SimpleITK nibabel numpy pandas matplotlib
```

Install additional packages only when their corresponding modules are
implemented. Keep a pinned `requirements.txt` for reproducibility.

## How to Run

Run commands from the project root in the VS Code PowerShell terminal.

### Convert DICOM to NIfTI

``` powershell
python preprocessing/dicom_to_nifti.py
```

### Run MRI quality checks

``` powershell
python preprocessing/image_preprocessing.py
```

### Verify generated NIfTI files

``` powershell
Get-ChildItem "dataset\nifti" -Recurse -File
```

### Inspect reports and visualizations

-   Conversion report: `results/dicom_conversion_report.csv`
-   QC reports: `results/preprocessing_qc_report_*.csv`
-   Slice plots: `results/plots/`

Training commands will be added after the mask pipeline and training
scripts are implemented and verified.

## Expected Outputs

  Output                        Description
  ----------------------------- ----------------------------------------------------
  `.nii.gz` volumes             Converted MRI series
  Conversion CSV                Per-series conversion status and metadata
  QC CSV                        Validation results for MRI volumes
  QC slice plots                Axial, coronal, and sagittal views
  Ground-truth masks            Reference hippocampus labels (pending)
  Model checkpoints             Trained segmentation weights (pending)
  Evaluation tables and plots   Held-out performance results (pending)
  Web reports                   Application-generated research summaries (planned)

## Evaluation Plan

### Segmentation

-   Dice similarity coefficient
-   Intersection over Union (IoU)
-   Precision
-   Recall / sensitivity
-   Specificity where appropriate
-   Visual inspection of overlays
-   Per-subject results and error analysis

### Classification (if implemented)

-   Accuracy
-   Precision, recall, and F1-score
-   Confusion matrix
-   ROC-AUC where appropriate
-   Subject-level cross-validation and held-out testing
-   Calibration and class-specific performance when feasible

Metrics should be calculated on appropriate held-out data. Do not
compare results from different data splits as if they were directly
equivalent.

## Team Responsibilities

  -----------------------------------------------------------------------
  Member                              Primary responsibility
  ----------------------------------- -----------------------------------
  Member 1                            Dataset collection, DICOM
                                      conversion, MRI preprocessing, QC,
                                      and data organization

  Member 2                            Ground-truth masks, dataset
                                      preparation, U-Net architecture,
                                      and training

  Member 3                            Segmentation evaluation, feature
                                      extraction, classification
                                      experiments, and analysis

  Member 4                            Frontend, backend, visualization
                                      integration, reports, deployment,
                                      and documentation
  -----------------------------------------------------------------------

All members should participate in integration, testing, literature
review, and final presentation. Work allocation and completion
percentages should be updated using verified deliverables.

## Reproducibility and Data Safety

-   Preserve original DICOM files; write converted and processed data to
    separate directories.
-   Do not commit restricted ADNI data, credentials, private keys, or
    personal information to GitHub.
-   Follow ADNI access and data-use requirements.
-   Record software versions, model configurations, random seeds, and
    dataset split identifiers.
-   Use subject-level splits to prevent leakage between training and
    evaluation.
-   Keep ground-truth labels separate from predictions.
-   Validate spatial alignment before calculating metrics or volumes.
-   Use secure storage and access controls for any uploaded medical
    images in the application.

## Limitations

-   The current verified dataset subset contains only five volumes.
-   Ground-truth hippocampus masks have not yet been confirmed in the
    current workflow.
-   No U-Net training or segmentation performance has been reported yet.
-   Classification feasibility depends on suitable labels and adequate
    data.
-   The application and deployment are planned, not yet verified.
-   Research outputs must not be presented as clinical diagnoses.

## Future Enhancements

-   Attention U-Net and 3D U-Net comparisons.
-   Robust longitudinal analysis for subjects with multiple visits.
-   3D segmentation visualization.
-   Model explainability and uncertainty analysis.
-   Automated PDF reports.
-   Secure user accounts and analysis history.
-   Docker-based deployment and system monitoring.
-   External validation on an independent dataset, if available and
    permitted.

## Acknowledgements

We acknowledge the Alzheimer's Disease Neuroimaging Initiative (ADNI)
for providing research data, subject to its data access and use
policies. Publications, datasets, and software used in the final project
should be cited in the project report.
