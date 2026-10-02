"""
Step 3: DICOM-to-NIfTI Conversion
==================================
Converts raw ADNI DICOM series into compressed NIfTI (.nii.gz) volumes.

Pipeline
--------
1. Recursively discover all .dcm files under DICOM_ROOT.
2. Group slices by SeriesInstanceUID (one series = one 3-D volume).
3. Extract subject ID from the ADNI directory hierarchy or DICOM tags.
4. Sort slices spatially using ImagePositionPatient (z-coordinate).
5. Reconstruct each 3-D volume with SimpleITK (preserves spacing, origin,
   and direction cosines).
6. Save as <SubjectID>_<SeriesUID_short>.nii.gz in NIFTI_OUTPUT_DIR.
7. Validate every output file with NiBabel.
8. Write a per-series CSV report to REPORT_PATH.

Usage
-----
    .venv\Scripts\python.exe preprocessing/dicom_to_nifti.py

No arguments are required; all paths are configured in the CONFIGURATION block
below.  The script never touches or deletes the original DICOM files.
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import csv
import logging
import os
import re
import sys
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
try:
    import pydicom
    import SimpleITK as sitk
    import nibabel as nib
    import numpy as np
except ImportError as exc:
    sys.exit(
        f"[FATAL] Missing dependency: {exc}\n"
        "Activate the project virtual environment and install requirements."
    )

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  (edit only this block if paths change)
# ════════════════════════════════════════════════════════════════════════════

# Root of the raw ADNI DICOM dataset
DICOM_ROOT = Path(r"D:\ADNI_T1_MRI_Baseline\ADNI")

# Where converted NIfTI files will be saved  (relative to this script's CWD
# so the command in the docstring works without extra flags)
NIFTI_OUTPUT_DIR = Path("dataset/nifti")

# Where the conversion report is written
REPORT_PATH = Path("results/dicom_conversion_report.csv")

# Minimum slices required to consider a series a valid 3-D volume
MIN_SLICES = 10

# Whether to overwrite an already-converted NIfTI file
OVERWRITE_EXISTING = False

# ════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ════════════════════════════════════════════════════════════════════════════

# Force UTF-8 output on Windows (prevents cp1252 UnicodeEncodeError)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════════════════════════════════


def _safe_tag(ds: "pydicom.Dataset", tag: str, default: str = "") -> str:
    """Return a DICOM tag value as a stripped string, or *default* on error."""
    try:
        val = getattr(ds, tag, None)
        return str(val).strip() if val is not None else default
    except Exception:
        return default


def _extract_subject_id_from_path(dcm_path: Path) -> str:
    """
    Derive the ADNI subject ID from the file path.

    ADNI directory layout:
        <root>/<SubjectID>/MP-RAGE/<Date>/<SeriesFolder>/<file>.dcm
                ^^^^^^^^^^
    The subject folder matches the pattern NNN_S_NNNN (e.g. 005_S_0324).
    Falls back to DICOM PatientID tag if directory matching fails.
    """
    adni_pattern = re.compile(r"\d{3}_S_\d{4}")
    for part in dcm_path.parts:
        if adni_pattern.fullmatch(part):
            return part
    return ""


def _extract_subject_id_from_dicom(ds: "pydicom.Dataset") -> str:
    """Return PatientID from DICOM header, sanitising for use as a filename."""
    pid = _safe_tag(ds, "PatientID")
    # Replace any character that is not alphanumeric, dash, or underscore
    return re.sub(r"[^\w\-]", "_", pid) if pid else "UNKNOWN"


def _image_position_z(ds: "pydicom.Dataset") -> float:
    """
    Return the z-component of ImagePositionPatient for slice ordering.
    Falls back to InstanceNumber, then SliceLocation, then 0.
    """
    try:
        ipp = ds.ImagePositionPatient
        return float(ipp[2])
    except Exception:
        pass
    try:
        return float(ds.InstanceNumber)
    except Exception:
        pass
    try:
        return float(ds.SliceLocation)
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 + 2: DISCOVERY AND GROUPING
# ════════════════════════════════════════════════════════════════════════════


def discover_and_group_dicom_files(dicom_root: Path) -> dict:
    """
    Recursively walk *dicom_root*, read every .dcm file, and group files by
    SeriesInstanceUID.

    Returns
    -------
    dict[series_uid -> {"files": [...], "subject_id": str}]
    """
    log.info("Scanning for DICOM files under: %s", dicom_root)

    series_map: dict = defaultdict(lambda: {"files": [], "subject_id": ""})
    total_files = 0
    skipped = 0

    for root, _dirs, files in os.walk(dicom_root):
        for fname in files:
            if not fname.lower().endswith(".dcm"):
                continue
            dcm_path = Path(root) / fname
            total_files += 1

            try:
                # Read only the header (stop_before_pixels) for speed; we will
                # re-read pixel data later through SimpleITK.
                ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            except Exception as exc:
                log.warning("Cannot read DICOM header [%s]: %s", dcm_path.name, exc)
                skipped += 1
                continue

            series_uid = _safe_tag(ds, "SeriesInstanceUID") or "NO_UID"

            # Subject ID: prefer directory-based extraction (more reliable)
            subject_id = _extract_subject_id_from_path(dcm_path)
            if not subject_id:
                subject_id = _extract_subject_id_from_dicom(ds)

            entry = series_map[series_uid]
            entry["files"].append(dcm_path)
            if not entry["subject_id"]:
                entry["subject_id"] = subject_id

    log.info(
        "Found %d .dcm files -> %d unique series  (%d files skipped on header read)",
        total_files,
        len(series_map),
        skipped,
    )
    return dict(series_map)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 + 4: SORT SLICES SPATIALLY
# ════════════════════════════════════════════════════════════════════════════


def sort_slices_spatially(file_list: list) -> list:
    """
    Sort a list of DICOM paths by their ImagePositionPatient z-coordinate
    (ascending).  Uses InstanceNumber and SliceLocation as fallbacks.

    Returns the sorted list.  Slices with identical z-values are sub-sorted
    by filename to ensure reproducibility.
    """

    def sort_key(dcm_path: Path):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
            return (_image_position_z(ds), str(dcm_path))
        except Exception:
            return (0.0, str(dcm_path))

    return sorted(file_list, key=sort_key)


# ════════════════════════════════════════════════════════════════════════════
# STEP 5: RECONSTRUCT 3-D VOLUME WITH SimpleITK
# ════════════════════════════════════════════════════════════════════════════


def build_volume_with_simpleitk(sorted_dcm_paths: list) -> "sitk.Image":
    """
    Use SimpleITK's ImageSeriesReader to build a 3-D ITK Image from a list
    of DICOM slice paths.  SimpleITK handles spacing, origin, and direction
    internally from the DICOM metadata.

    Parameters
    ----------
    sorted_dcm_paths : list[Path]
        Spatially-sorted DICOM file paths belonging to ONE series.

    Returns
    -------
    sitk.Image  — a 3-D image with correct voxel spacing, origin, and
                  direction cosines.
    """
    reader = sitk.ImageSeriesReader()
    # Cast to str list (SimpleITK does not accept Path objects)
    file_names = [str(p) for p in sorted_dcm_paths]
    reader.SetFileNames(file_names)
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    image = reader.Execute()
    return image


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 + 7: SAVE AS COMPRESSED NIfTI
# ════════════════════════════════════════════════════════════════════════════


def save_nifti(sitk_image: "sitk.Image", output_path: Path) -> None:
    """
    Write an ITK image to a compressed NIfTI file (.nii.gz).

    SimpleITK's WriteImage preserves voxel spacing, origin, and the full
    3×3 direction-cosine matrix as embedded in the ITK image object.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk_image, str(output_path), useCompression=True)


# ════════════════════════════════════════════════════════════════════════════
# STEP 8: VALIDATE WITH NiBabel
# ════════════════════════════════════════════════════════════════════════════


def validate_nifti(nifti_path: Path) -> tuple:
    """
    Load the saved NIfTI file with NiBabel and perform basic sanity checks.

    Returns
    -------
    (is_valid: bool, dimensions: tuple, voxel_spacing: tuple, error: str)
    """
    try:
        img = nib.load(str(nifti_path))
        shape = img.shape           # (X, Y, Z) or (X, Y, Z, T)
        zooms = img.header.get_zooms()  # voxel dimensions in mm

        # Basic checks
        if len(shape) < 3:
            return False, shape, zooms, "Less than 3 dimensions"
        if any(s == 0 for s in shape):
            return False, shape, zooms, "Zero-size dimension detected"

        data = img.get_fdata()
        if np.all(data == 0):
            return False, shape, zooms, "Volume contains only zeros"

        return True, shape, zooms, ""

    except Exception as exc:
        return False, (), (), str(exc)


# ════════════════════════════════════════════════════════════════════════════
# OUTPUT FILENAME BUILDER
# ════════════════════════════════════════════════════════════════════════════


def build_output_filename(subject_id: str, series_uid: str) -> str:
    """
    Construct a safe, human-readable output filename.

    Format: <SubjectID>_<last16chars_of_SeriesUID>.nii.gz
    """
    # Use the trailing 16 characters of the UID to keep the name manageable
    uid_short = series_uid[-16:].replace(".", "_")
    safe_sid = re.sub(r"[^\w\-]", "_", subject_id) if subject_id else "UNKNOWN"
    return f"{safe_sid}_{uid_short}.nii.gz"


# ════════════════════════════════════════════════════════════════════════════
# STEP 9 + 10: CSV REPORT
# ════════════════════════════════════════════════════════════════════════════

REPORT_FIELDNAMES = [
    "subject_id",
    "series_uid",
    "num_slices",
    "dimensions",
    "voxel_spacing_mm",
    "output_filename",
    "status",
    "error",
]


def init_report_writer(report_path: Path):
    """Open the CSV report file and return (file_handle, csv.DictWriter)."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(report_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=REPORT_FIELDNAMES)
    writer.writeheader()
    return fh, writer


def write_report_row(
    writer: "csv.DictWriter",
    subject_id: str,
    series_uid: str,
    num_slices: int,
    dimensions: tuple,
    voxel_spacing: tuple,
    output_filename: str,
    status: str,
    error: str = "",
) -> None:
    writer.writerow(
        {
            "subject_id": subject_id,
            "series_uid": series_uid,
            "num_slices": num_slices,
            "dimensions": "x".join(str(d) for d in dimensions) if dimensions else "",
            "voxel_spacing_mm": (
                " x ".join(f"{v:.4f}" for v in voxel_spacing) if voxel_spacing else ""
            ),
            "output_filename": output_filename,
            "status": status,
            "error": error,
        }
    )


# ════════════════════════════════════════════════════════════════════════════
# MAIN CONVERSION LOOP
# ════════════════════════════════════════════════════════════════════════════


def convert_all_series(
    dicom_root: Path,
    output_dir: Path,
    report_path: Path,
    min_slices: int = MIN_SLICES,
    overwrite: bool = OVERWRITE_EXISTING,
) -> None:
    """
    Orchestrate the full DICOM -> NIfTI conversion pipeline for every series
    discovered under *dicom_root*.
    """
    # ── counters ─────────────────────────────────────────────────────────
    n_success = 0
    n_skipped_existing = 0
    n_skipped_few_slices = 0
    n_failed = 0

    # ── discover + group ─────────────────────────────────────────────────
    series_map = discover_and_group_dicom_files(dicom_root)

    # ── open report ──────────────────────────────────────────────────────
    report_fh, report_writer = init_report_writer(report_path)

    total_series = len(series_map)
    log.info("Processing %d series …", total_series)

    try:
        for idx, (series_uid, info) in enumerate(series_map.items(), start=1):
            subject_id = info["subject_id"]
            files = info["files"]
            num_slices = len(files)
            out_fname = build_output_filename(subject_id, series_uid)
            out_path = output_dir / out_fname

            log.info(
                "[%d/%d] Subject %-14s  SeriesUID …%-16s  slices=%d",
                idx,
                total_series,
                subject_id,
                series_uid[-16:],
                num_slices,
            )

            # ── guard: too few slices ─────────────────────────────────
            if num_slices < min_slices:
                reason = f"Only {num_slices} slice(s) — minimum is {min_slices}"
                log.warning("  SKIPPED (too few slices): %s", reason)
                n_skipped_few_slices += 1
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    (), (), out_fname, "SKIPPED_FEW_SLICES", reason,
                )
                continue

            # ── guard: already converted ─────────────────────────────
            if out_path.exists() and not overwrite:
                log.info("  SKIPPED (already exists): %s", out_fname)
                n_skipped_existing += 1
                # Still validate existing file to fill in dimensions/spacing
                _valid, _dims, _spacing, _err = validate_nifti(out_path)
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    _dims, _spacing, out_fname,
                    "SKIPPED_EXISTS" if _valid else "SKIPPED_EXISTS_INVALID",
                    _err,
                )
                continue

            # ── sort slices spatially (Step 4) ────────────────────────
            try:
                sorted_files = sort_slices_spatially(files)
            except Exception as exc:
                err = f"Slice sorting failed: {exc}"
                log.error("  FAILED: %s", err)
                n_failed += 1
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    (), (), out_fname, "FAILED", err,
                )
                continue

            # ── reconstruct 3-D volume (Step 5) ──────────────────────
            try:
                sitk_image = build_volume_with_simpleitk(sorted_files)
            except Exception as exc:
                err = f"SimpleITK reconstruction failed: {exc}"
                log.error("  FAILED: %s", err)
                n_failed += 1
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    (), (), out_fname, "FAILED", err,
                )
                continue

            # ── save NIfTI (Steps 6 + 7) ──────────────────────────────
            try:
                save_nifti(sitk_image, out_path)
            except Exception as exc:
                err = f"NIfTI write failed: {exc}"
                log.error("  FAILED: %s", err)
                n_failed += 1
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    (), (), out_fname, "FAILED", err,
                )
                continue

            # ── validate with NiBabel (Step 8) ────────────────────────
            is_valid, dims, spacing, val_err = validate_nifti(out_path)
            if not is_valid:
                # Remove the invalid file so a rerun can attempt again
                try:
                    out_path.unlink(missing_ok=True)
                except Exception:
                    pass
                err = f"NiBabel validation failed: {val_err}"
                log.error("  FAILED validation: %s", err)
                n_failed += 1
                write_report_row(
                    report_writer, subject_id, series_uid, num_slices,
                    dims, spacing, out_fname, "FAILED_VALIDATION", err,
                )
                continue

            # ── success ───────────────────────────────────────────────
            n_success += 1
            log.info(
                "  OK  dims=%s  spacing=%s mm  -> %s",
                "x".join(str(d) for d in dims),
                " x ".join(f"{v:.3f}" for v in spacing),
                out_fname,
            )
            write_report_row(
                report_writer, subject_id, series_uid, num_slices,
                dims, spacing, out_fname, "SUCCESS", "",
            )

    finally:
        report_fh.close()

    # ── final summary ─────────────────────────────────────────────────────
    print()
    print("=" * 65)
    print("  DICOM -> NIfTI CONVERSION SUMMARY")
    print("=" * 65)
    print(f"  Total series discovered   : {total_series}")
    print(f"  Successfully converted    : {n_success}")
    print(f"  Skipped (already exist)   : {n_skipped_existing}")
    print(f"  Skipped (too few slices)  : {n_skipped_few_slices}")
    print(f"  Failed                    : {n_failed}")
    print(f"  NIfTI output directory    : {output_dir.resolve()}")
    print(f"  Conversion report         : {report_path.resolve()}")
    print("=" * 65)

    if n_failed:
        log.warning(
            "%d series failed — review '%s' for details.", n_failed, report_path
        )
    else:
        log.info("All series converted successfully.")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log.info("=" * 65)
    log.info("  Step 3: DICOM -> NIfTI Conversion")
    log.info("  Started : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("  DICOM root : %s", DICOM_ROOT)
    log.info("  Output dir : %s", NIFTI_OUTPUT_DIR.resolve())
    log.info("  Report     : %s", REPORT_PATH.resolve())
    log.info("=" * 65)

    # Validate DICOM root
    if not DICOM_ROOT.exists():
        sys.exit(
            f"[FATAL] DICOM_ROOT does not exist: {DICOM_ROOT}\n"
            "Edit the CONFIGURATION block at the top of this script."
        )

    # Ensure output directory exists
    NIFTI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    convert_all_series(
        dicom_root=DICOM_ROOT,
        output_dir=NIFTI_OUTPUT_DIR,
        report_path=REPORT_PATH,
        min_slices=MIN_SLICES,
        overwrite=OVERWRITE_EXISTING,
    )


if __name__ == "__main__":
    main()
