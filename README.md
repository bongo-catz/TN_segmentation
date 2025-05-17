# TN_segmentation

This repository contains code for multi-stage rigid and non-rigid registration of MRI and MRA neuroimaging scans, with a specific focus on localized alignment around the trigeminal nerve using centroid-based region selection and SimpleITK-based transformations.

The pipeline supports preprocessing (including N4 bias field correction and intensity normalization), affine/B-spline registration, mask generation from annotated centroids, and visualization of results for manual inspection and region confirmation.

## Repository Structure
```
TN_segmentation/
├── requirements.txt # Python dependencies
├── environment.yml # Conda environment file
├── README.md
├── scripts/
│ ├── alignment.py # Base alignment script (v1)
│ ├── alignment_v2.py # Further automated script
│ ├── helper_class/
│ │ ├── region_selector.py # ROI selector class
│ │ ├── region_viewer.py # ROI viewer class
├── output/ # Output results and logs (e.g. aligned MRA)
│ ├── <patient_id>/
│ │ ├── aligned_MRA.nii.gz
│ │ ├── alignment.log
├── data/ # Input image and metadata storage
│ ├── centroids/ # CSV files with annotated centroid locations
│ │ ├── <year>_centroids.csv
│ ├── <patient_id>/ # Patient-specific folders
│ │ ├── <patient_id>_MRI/
│ │ ├── <patient_id>_MRA/
```

## Setup

### 1. Install via Conda

```bash
conda env create -f environment.yml
conda activate tn_seg
```

### 2. Install via pip
```
pip install -r requirements.txt
```

## Usage

```python
python scripts/alignment_v2.py \
  --mri data/<patient_id>/<patient_id>_MRI/nifti/<patient_id>_MRI.nii.gz \
  --mra data/<patient_id>/<patient_id>_MRA/nifti/<patient_id>_MRA.nii.gz \
  --output output/<patient_id>/aligned_MRA.nii.gz \
  --patient-mrn <patient_id> \
  --initial-transform euler \
  --initial-spacing 50.0 \
  --final-spacing 30.0 \
  --centroid-locations data/centroids/<year>_centroids.csv \
  --logfile output/<patient_id>/alignment.log
```
