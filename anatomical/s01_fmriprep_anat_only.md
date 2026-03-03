# Fmriprep
- check you have installed everything correctly (following s00_anat_pipeline.md)
- Pipeline assumes one set of anatomies
- for s01_fmriprep_anat_only.sh; just pass the BIDS directory; subject id; & name of the session with the anatomy inside 
- This script will create a "fake" fmriprep session (fprep) to put all of our preprocessed stuff inside later
- For now just doing the anatomy

```bash
# path to BIDS directory
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
bash s01_fmriprep_anat_only.sh --bids_dir $BIDS_DIR --sub 01 --ses 01
```