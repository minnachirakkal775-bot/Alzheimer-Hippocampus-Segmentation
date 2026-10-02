"""
preprocessing/image_preprocessing.py
=====================================
Step 4: MRI Preprocessing and Quality Checking
-----------------------------------------------
Alzheimer's Disease Detection – Hippocampus Segmentation with U-Net

Purpose
-------
* Scan dataset/nifti/ for every .nii / .nii.gz volume.
* Perform quality checks (readability, shape, NaN/Inf, empty volumes,
  intensity outliers).
* Preserve original geometry and spatial metadata — NO resampling or
  arbitrary resizing at this stage.
* Copy valid volumes to dataset/preprocessed/ (never overwrite without
  confirmation; never touch the originals).
* Save a CSV quality-check report to results/.
* Save before/after slice visualisation panels to results/plots/.
* Print a processing summary to the console.

Usage (from project root)
--------------------------
    python preprocessing/image_preprocessing.py

    # Quiet mode – suppress individual file logs:
    python preprocessing/image_preprocessing.py --quiet

    # Custom input / output paths:
    python preprocessing/image_preprocessing.py \
        --input  dataset/nifti \
        --output dataset/preprocessed \
        --results results
"""

import argparse
import logging
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe in all environments
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Configure and return a named logger for this module."""
    logger = logging.getLogger("mri_preprocessing")
    if not logger.handlers:
        level = logging.DEBUG if verbose else logging.WARNING
        # Force UTF-8 on Windows to prevent cp1252 encoding errors
        stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_nifti_files(input_dir: Path) -> List[Path]:
    """
    Recursively collect all .nii and .nii.gz files in *input_dir*.

    Returns a sorted list of absolute Path objects.
    """
    patterns = ["**/*.nii", "**/*.nii.gz"]
    found: List[Path] = []
    for pattern in patterns:
        found.extend(input_dir.glob(pattern))
    # deduplicate and sort
    return sorted(set(found))


# ---------------------------------------------------------------------------
# NIfTI loading helpers (NiBabel)
# ---------------------------------------------------------------------------

def load_nifti_nibabel(
    filepath: Path, logger: logging.Logger
) -> Tuple[Optional[nib.Nifti1Image], Optional[str]]:
    """
    Attempt to load a NIfTI file with NiBabel.

    Returns
    -------
    (image, None)       on success
    (None, error_msg)   on failure
    """
    try:
        img = nib.load(str(filepath))
        # Trigger actual data load to catch decompression errors early
        _ = img.get_fdata(dtype=np.float32)
        return img, None
    except Exception as exc:
        return None, f"NiBabel load error: {exc}"


# ---------------------------------------------------------------------------
# SimpleITK loading helper
# ---------------------------------------------------------------------------

def load_nifti_sitk(
    filepath: Path, logger: logging.Logger
) -> Tuple[Optional[sitk.Image], Optional[str]]:
    """
    Attempt to load a NIfTI file with SimpleITK (cross-check).

    Returns
    -------
    (sitk_image, None)  on success
    (None, error_msg)   on failure
    """
    try:
        img = sitk.ReadImage(str(filepath))
        return img, None
    except Exception as exc:
        return None, f"SimpleITK load error: {exc}"


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(nib_img: nib.Nifti1Image) -> Dict:
    """
    Extract spatial and header metadata from a loaded NiBabel image.

    Returns a dict with keys:
        dimensions, voxel_spacing, origin, orientation_code,
        affine_det, data_dtype
    """
    header = nib_img.header
    affine = nib_img.affine
    shape = nib_img.shape
    zooms = header.get_zooms()

    # Orientation string (e.g., 'RAS', 'LAS')
    try:
        orientation_code = "".join(nib.aff2axcodes(affine))
    except Exception:
        orientation_code = "UNKNOWN"

    # Origin = translation column of the affine
    origin = tuple(float(x) for x in affine[:3, 3])

    # Determinant of the affine 3×3 submatrix (sign encodes handedness)
    affine_det = float(np.linalg.det(affine[:3, :3]))

    return {
        "dimensions": tuple(int(d) for d in shape),
        "voxel_spacing": tuple(float(z) for z in zooms[:3]),
        "origin": origin,
        "orientation_code": orientation_code,
        "affine_det": affine_det,
        "data_dtype": str(header.get_data_dtype()),
    }


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def check_volume_quality(
    data: np.ndarray,
    metadata: Dict,
    logger: logging.Logger,
) -> Tuple[bool, str, Dict]:
    """
    Run a battery of quality checks on the voxel array.

    Parameters
    ----------
    data      : float32 ndarray (already loaded via get_fdata)
    metadata  : dict from extract_metadata()
    logger    : module logger

    Returns
    -------
    (is_valid, status_message, stats_dict)
    """
    issues: List[str] = []
    stats: Dict = {}

    # -- Dimensionality ---------------------------------------------------
    ndim = data.ndim
    if ndim not in (3, 4):
        issues.append(f"Unexpected number of dimensions: {ndim} (expected 3 or 4)")

    # -- Empty volume -----------------------------------------------------
    total_voxels = data.size
    if total_voxels == 0:
        issues.append("Volume is empty (zero voxels)")
        stats.update({"min_intensity": None, "max_intensity": None,
                       "mean_intensity": None, "std_intensity": None,
                       "nan_count": 0, "inf_count": 0,
                       "nonzero_fraction": 0.0})
        return False, "; ".join(issues), stats

    # -- NaN / Inf --------------------------------------------------------
    nan_count = int(np.sum(np.isnan(data)))
    inf_count = int(np.sum(np.isinf(data)))
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN value(s)")
    if inf_count > 0:
        issues.append(f"Contains {inf_count} Inf value(s)")

    # Replace NaN/Inf for statistics only (NOT in the saved file)
    data_finite = np.where(np.isfinite(data), data, 0.0)

    # -- Intensity statistics ---------------------------------------------
    min_val = float(np.min(data_finite))
    max_val = float(np.max(data_finite))
    mean_val = float(np.mean(data_finite))
    std_val = float(np.std(data_finite))

    stats.update({
        "min_intensity": round(min_val, 4),
        "max_intensity": round(max_val, 4),
        "mean_intensity": round(mean_val, 4),
        "std_intensity": round(std_val, 4),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "nonzero_fraction": round(float(np.count_nonzero(data_finite)) / total_voxels, 4),
    })

    # -- Nearly-empty / all-zero volume -----------------------------------
    if max_val == 0.0:
        issues.append("All voxel intensities are zero (blank volume)")

    # -- Constant (no variation) volume -----------------------------------
    if std_val < 1e-8 and max_val != 0.0:
        issues.append(f"Volume has near-zero intensity variation (std={std_val:.2e})")

    # -- Extreme intensity outlier heuristic ------------------------------
    # Flag volumes where the 99th percentile is >100× the median
    # (indicative of a hot-spot artefact or wrong scaling).
    nonzero = data_finite[data_finite > 0]
    if nonzero.size > 0:
        median_nz = float(np.median(nonzero))
        p99_nz = float(np.percentile(nonzero, 99))
        if median_nz > 0 and p99_nz / median_nz > 100:
            issues.append(
                f"Potential intensity outlier: p99={p99_nz:.1f}, median={median_nz:.1f}"
            )

    # -- Voxel spacing sanity (warn, not reject) --------------------------
    spacing = metadata.get("voxel_spacing", (1.0, 1.0, 1.0))
    if any(s <= 0 for s in spacing):
        issues.append(f"Non-positive voxel spacing detected: {spacing}")

    is_valid = len(issues) == 0
    status = "PASS" if is_valid else ("WARN" if (nan_count > 0 or inf_count > 0) else "FAIL")

    # Treat NaN/Inf as warnings but still reject for safety
    if nan_count > 0 or inf_count > 0:
        is_valid = False

    return is_valid, status + (": " + "; ".join(issues) if issues else ""), stats


# ---------------------------------------------------------------------------
# Copy / save (preserve original)
# ---------------------------------------------------------------------------

def save_preprocessed_volume(
    src: Path,
    dst_dir: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> Tuple[bool, str]:
    """
    Copy *src* into *dst_dir*, preserving the original file untouched.

    Returns (success, message).
    """
    dst = dst_dir / src.name

    if dst.exists() and not overwrite:
        logger.warning("  SKIP  %s already exists in output folder.", src.name)
        return False, f"Output file already exists: {dst.name} (use --overwrite to replace)"

    try:
        shutil.copy2(str(src), str(dst))
        logger.debug("  COPY  %s -> %s", src.name, dst_dir)
        return True, "Copied successfully"
    except OSError as exc:
        return False, f"Copy failed: {exc}"


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _mid_slices(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the central axial, coronal, and sagittal slices."""
    # Ensure 3-D (take first volume if 4-D)
    vol = data[..., 0] if data.ndim == 4 else data
    ax = int(vol.shape[0] // 2)
    co = int(vol.shape[1] // 2)
    sa = int(vol.shape[2] // 2)
    return vol[ax, :, :], vol[:, co, :], vol[:, :, sa]


def _plot_three_views(
    data: np.ndarray,
    title: str,
    ax_row: List,
    cmap: str = "gray",
) -> None:
    """
    Plot axial / coronal / sagittal slices into *ax_row* (list of 3 Axes).
    """
    slices = _mid_slices(data)
    view_labels = ["Axial", "Coronal", "Sagittal"]
    for ax, slc, label in zip(ax_row, slices, view_labels):
        ax.imshow(np.rot90(slc), cmap=cmap, aspect="auto")
        ax.set_title(f"{title}\n{label}", fontsize=8)
        ax.axis("off")


def generate_visualisation(
    nib_img: nib.Nifti1Image,
    filename: str,
    plots_dir: Path,
    logger: logging.Logger,
) -> Optional[Path]:
    """
    Create a 2-row × 3-column figure showing raw ("before") and the same
    data ("after" — demonstrating that geometry is preserved) and save it
    to *plots_dir*.

    Because this step explicitly does NOT apply any intensity transformation,
    the "after" panel shows the same image with a clearly labelled note.

    Returns the output Path, or None on error.
    """
    try:
        data = nib_img.get_fdata(dtype=np.float32)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        fig.suptitle(
            f"MRI Quality Check – {filename}\n"
            "(No intensity transformation applied at preprocessing stage)",
            fontsize=10,
            fontweight="bold",
        )

        # Row 0 – "Before" (raw loaded data)
        _plot_three_views(data, "Before (raw)", axes[0].tolist())

        # Row 1 – "After" (same file, confirming geometry is preserved)
        # Clip to [1st, 99th] percentile for visual clarity only — the saved
        # NIfTI file is identical to the original.
        finite = data[np.isfinite(data)]
        if finite.size > 0:
            lo, hi = np.percentile(finite, [1, 99])
            data_display = np.clip(data, lo, hi)
        else:
            data_display = data

        _plot_three_views(data_display, "After (geometry preserved, display clipped)", axes[1].tolist())

        plt.tight_layout()
        stem = Path(filename).stem  # removes .nii or .nii.gz
        if stem.endswith(".nii"):
            stem = stem[:-4]        # handle .nii.gz case
        out_path = plots_dir / f"{stem}_qc_slices.png"
        plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.debug("  PLOT  %s", out_path.name)
        return out_path

    except Exception as exc:
        logger.warning("  Could not generate plot for %s: %s", filename, exc)
        plt.close("all")
        return None


# ---------------------------------------------------------------------------
# Per-file processing pipeline
# ---------------------------------------------------------------------------

def process_single_volume(
    filepath: Path,
    output_dir: Path,
    plots_dir: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> Dict:
    """
    Full pipeline for a single NIfTI volume:
        1. Load (NiBabel + SimpleITK cross-check)
        2. Extract metadata
        3. Quality checks
        4. Copy valid volume to output_dir
        5. Generate visualisation

    Returns a record dict suitable for appending to the CSV report.
    """
    record: Dict = {
        "filename": filepath.name,
        "filepath": str(filepath),
        "dimensions": None,
        "voxel_spacing": None,
        "origin": None,
        "orientation": None,
        "min_intensity": None,
        "max_intensity": None,
        "mean_intensity": None,
        "std_intensity": None,
        "nan_count": None,
        "inf_count": None,
        "nonzero_fraction": None,
        "affine_det": None,
        "data_dtype": None,
        "qc_status": "PENDING",
        "error_message": "",
        "saved_to_output": False,
        "plot_saved": False,
    }

    logger.info("Processing: %s", filepath.name)

    # ── 1. Load with NiBabel ──────────────────────────────────────────────
    nib_img, nib_err = load_nifti_nibabel(filepath, logger)
    if nib_img is None:
        logger.error("  [FAIL] %s", nib_err)
        record["qc_status"] = "FAIL"
        record["error_message"] = nib_err
        return record

    # ── 2. Cross-check with SimpleITK ─────────────────────────────────────
    sitk_img, sitk_err = load_nifti_sitk(filepath, logger)
    if sitk_img is None:
        logger.warning("  [WARN] SimpleITK could not read file: %s", sitk_err)
        record["error_message"] = f"SimpleITK warning: {sitk_err}"

    # ── 3. Extract metadata ───────────────────────────────────────────────
    try:
        meta = extract_metadata(nib_img)
    except Exception as exc:
        record["qc_status"] = "FAIL"
        record["error_message"] = f"Metadata extraction failed: {exc}"
        logger.error("  [FAIL] %s", record["error_message"])
        return record

    record.update({
        "dimensions":    str(meta["dimensions"]),
        "voxel_spacing": str(meta["voxel_spacing"]),
        "origin":        str(meta["origin"]),
        "orientation":   meta["orientation_code"],
        "affine_det":    round(meta["affine_det"], 4),
        "data_dtype":    meta["data_dtype"],
    })

    # ── 4. Load voxel data ────────────────────────────────────────────────
    try:
        data = nib_img.get_fdata(dtype=np.float32)
    except Exception as exc:
        record["qc_status"] = "FAIL"
        record["error_message"] = f"Data access failed: {exc}"
        logger.error("  [FAIL] %s", record["error_message"])
        return record

    # ── 5. Quality checks ─────────────────────────────────────────────────
    is_valid, status_msg, stats = check_volume_quality(data, meta, logger)

    record.update(stats)
    record["qc_status"] = status_msg

    if is_valid:
        logger.info("  [PASS] dim=%s  spacing=%s", meta["dimensions"], meta["voxel_spacing"])
    else:
        logger.warning("  [%s] %s", "FAIL", status_msg)
        record["error_message"] = status_msg

    # ── 6. Copy valid volume (preserve original) ───────────────────────────
    if is_valid:
        saved, save_msg = save_preprocessed_volume(filepath, output_dir, overwrite, logger)
        record["saved_to_output"] = saved
        if not saved:
            record["error_message"] = save_msg

    # ── 7. Visualisation ──────────────────────────────────────────────────
    plot_path = generate_visualisation(nib_img, filepath.name, plots_dir, logger)
    record["plot_saved"] = plot_path is not None

    return record


# ---------------------------------------------------------------------------
# CSV report
# ---------------------------------------------------------------------------

def save_report(records: List[Dict], results_dir: Path, logger: logging.Logger) -> Path:
    """
    Build a Pandas DataFrame from *records* and write it as a CSV file
    to *results_dir*.

    Returns the CSV path.
    """
    df = pd.DataFrame(records)

    # Friendly column order
    col_order = [
        "filename", "dimensions", "voxel_spacing", "origin", "orientation",
        "data_dtype", "affine_det",
        "min_intensity", "max_intensity", "mean_intensity", "std_intensity",
        "nan_count", "inf_count", "nonzero_fraction",
        "qc_status", "error_message",
        "saved_to_output", "plot_saved", "filepath",
    ]
    # Keep only columns that exist (defensive)
    col_order = [c for c in col_order if c in df.columns]
    df = df[col_order]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"preprocessing_qc_report_{timestamp}.csv"
    df.to_csv(str(csv_path), index=False)
    logger.info("Report saved -> %s", csv_path)
    return csv_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(records: List[Dict], csv_path: Path) -> None:
    """Print a concise processing summary to stdout."""
    total = len(records)
    passed = sum(1 for r in records if r.get("saved_to_output"))
    failed = sum(1 for r in records if "FAIL" in str(r.get("qc_status", "")))
    warned = sum(1 for r in records if "WARN" in str(r.get("qc_status", "")))
    skipped = total - passed - failed - warned

    sep = "=" * 60
    print(f"\n{sep}")
    print("  MRI PREPROCESSING AND QUALITY CHECK - SUMMARY")
    print(sep)
    print(f"  Total volumes found   : {total}")
    print(f"  Valid  (saved)        : {passed}")
    print(f"  Warnings              : {warned}")
    print(f"  Rejected (failed QC)  : {failed}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Report saved to       : {csv_path}")
    print(sep + "\n")

    if failed > 0:
        print("  Failed volumes:")
        for r in records:
            if "FAIL" in str(r.get("qc_status", "")):
                print(f"    • {r['filename']}: {r.get('error_message', '')}")
        print()

    if total == 0:
        print(
            "  [INFO] No NIfTI files were found in the input directory.\n"
            "         Run the DICOM-to-NIfTI conversion step first, then\n"
            "         re-run this script.\n"
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 4: MRI Preprocessing & Quality Checking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", default="dataset/nifti",
        help="Input directory containing .nii / .nii.gz files.",
    )
    parser.add_argument(
        "--output", default="dataset/preprocessed",
        help="Output directory for valid preprocessed volumes.",
    )
    parser.add_argument(
        "--results", default="results",
        help="Root results directory (CSV report goes here).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-file DEBUG messages (show warnings/errors only).",
    )
    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_arg_parser().parse_args()
    logger = setup_logging(verbose=not args.quiet)

    # ── Resolve paths (relative to CWD or absolute) ─────────────────────
    project_root = Path.cwd()
    input_dir    = (project_root / args.input).resolve()
    output_dir   = (project_root / args.output).resolve()
    results_dir  = (project_root / args.results).resolve()
    plots_dir    = results_dir / "plots"

    # ── Create output directories if they don't exist ────────────────────
    for d in (output_dir, results_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger.info("Input  directory : %s", input_dir)
    logger.info("Output directory : %s", output_dir)
    logger.info("Results directory: %s", results_dir)
    logger.info("Plots  directory : %s", plots_dir)
    logger.info("Overwrite mode   : %s", args.overwrite)

    # ── Discover NIfTI files ─────────────────────────────────────────────
    nifti_files = find_nifti_files(input_dir)
    logger.info("Found %d NIfTI file(s) in %s", len(nifti_files), input_dir)

    if not nifti_files:
        print(
            f"\n[INFO] No .nii or .nii.gz files found in '{input_dir}'.\n"
            "       Please ensure the DICOM-to-NIfTI conversion (Step 3)\n"
            "       has been completed and files are placed in that folder.\n"
        )
        # Still write an empty report so the pipeline doesn't break downstream.
        csv_path = save_report([], results_dir, logger)
        print_summary([], csv_path)
        return

    # ── Process each volume ───────────────────────────────────────────────
    records: List[Dict] = []
    for filepath in nifti_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress nibabel header warnings
            record = process_single_volume(
                filepath, output_dir, plots_dir, args.overwrite, logger
            )
        records.append(record)

    # ── Save CSV report ───────────────────────────────────────────────────
    csv_path = save_report(records, results_dir, logger)

    # ── Print summary ─────────────────────────────────────────────────────
    print_summary(records, csv_path)


if __name__ == "__main__":
    main()
