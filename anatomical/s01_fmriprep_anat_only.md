# Fmriprep
How to install fmriprep docker quickly
- TODO prescribe a specific version. (25.2.4) ?
```bash
mamba create -n fmriprep001 python
conda activate fmriprep001 
python -m pip install fmriprep-docker
```

# Run fmriprep anat only
```bash
PROJ_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR="${PROJ_DIR}/derivatives/freesurfer"
bash s01_fmriprep_anat_only.sh --bids_dir $BIDS_DIR --sub hp01
```