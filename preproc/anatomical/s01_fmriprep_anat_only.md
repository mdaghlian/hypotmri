# Fmriprep
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















# run fmriprep 2 
fmriprep-docker \
  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/ \
  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/fmriprep \
  participant \
  --participant-label sub-hp01 \
  --fs-subjects-dir  /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/BIDS/derivatives/freesurfer \
  --output-spaces func T1w fsnative \
  --fs-license-file /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/code/license.txt \
  --skip-bids-validation \
  --omp-nthreads 8 \
  -w /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/BIDSWF