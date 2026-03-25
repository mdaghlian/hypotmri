# Anatomical pipeline: 
Assumes that you have already run the setup

Make sure you have updated project specific information in you /config/config_pipeline.sh file (i.e., correct paths etc)

## [1] fmriprep_anat_only
- pass the BIDS directory; subject id; & name of the session with the anatomy (T1w image) inside 
- This script will create a "fake" fmriprep session (fprep) to put all of our preprocessed stuff inside later
- For now just doing the anatomy

```bash
BIDS_DIR="/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/"
bash s01_fmriprep_anat_only.sh --bids-dir ${BIDS_DIR} --sub sub-hp01 --ses ses-01
```
to submit to the cluster with qsub run 
```bash
bash s01_hpc_submit.sh --bids-dir ${BIDS_DIR} --sub sub-hp01 --ses ses-01
```

*TODO* - add a QC here

## [2] benson atlas
Step [1] will create the subject specific freesurfer folder. Now we use this to create the benson atlas
```bash
conda activate b14 # python environment we need
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR=$BIDS_DIR/derivatives/freesurfer
python s02_b14atlas.py sub-hp01 --SUBJECTS_DIR $SUBJECTS_DIR
```
This will add benson atlas to your freesurfer folder and add the ROIS to your label folder 

---
## Following steps are optional - but may be useful for visualisation
## [3] autoflatten 
Makes cuts to your inflated freesurfer surface so that they can be easily used as flatmaps

```bash
conda activate autoflat
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR=$BIDS_DIR/derivatives/freesurfer
# run on a freesurfer subjects
autoflatten $SUBJECTS_DIR/sub-hp01
```
## [4] pycortex 
Takes the freesurfer + autoflatten outputs and makes a subject specific pycortex directory. this can be used for making quick flatmaps. It also automatically imports the freesurfer ROIs on top. 

```bash
conda activate pctx
python s04_pycortex.py sub-hp04 --fsdir $SUBJECTS_DIR
```