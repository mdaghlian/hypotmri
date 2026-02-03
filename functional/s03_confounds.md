# Confounds

Use fmriprep to generate confounds. 
In order to do that we need to feed fmriprep the nice "preproc" style data from previous sections. we create a "fake" session "fprep" which is what we tell it to runs  

```bash
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot
export SUBJECTS_DIR=$BIDS_DIR/derivatives/freesurfer
bash s03_confounds.sh --input_dir $BIDS_DIR/derivatives/sf2_coregAFNITOP/sub-hp01/ses-01 --bids_dir $BIDS_DIR --sub sub-hp01 
```