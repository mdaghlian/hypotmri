# Anatomical pipeline: 
Assumes that you have already followed the setup process ([local_setup.md](../config/local_setup.md))

Make sure you have updated project specific information in you /config/project_XXX.sh file (i.e., correct paths etc). Before running analyses make sure the correct project is set:

```bash
source set_project.sh XXX # (your project name)
```

## [1] fmriprep_anat_only
Runs fmriprep anatomy only, which, also does freesurfer as well
- pass the BIDS directory; subject id; & name of the session with the anatomy (T1w image) inside 
- This script will create a separate "FPREP_BIDS" folder inside derivatives. The reason for this is that we have our own preprocessing steps we want to run outside of fmriprep - by splitting it up like this, we can make a clean distinction between fmriprep-run processes & our own.

#### Run locally:
```bash
s01_fmriprep_anat_only.sh --bids-dir ${BIDS_DIR} --sub <sub-id> --ses <ses-id>
```
#### Run on HPC
Assumes you have already follwed the setup instructions in [hpc_setup.md](../config/hpc_setup.md)) 
```bash
s01_hpc_submit.sh --bids-dir ${BIDS_DIR} --sub <sub-id> --ses <ses-id>
```
To pull the freesurfer output afterwards to your local pc run:
```bash
rsync_from_hpc.sh --sub <sub-id> --ses <ses-id> --deriv freesurfer
```

#### Qualtiy Checks (local only)
```bash
# Open freeview with anatomy + wm + gm boundaryies
qc_surfaces.sh <sub-id> 

# Create movie (in $SUBJECTS_DIR/<sub-id>/movie)
# moving through the saggital view, with wm + gm boundaries
# good for spotting saggital sinus erros 
qc_surfaces_movie.sh <sub-id>
```

## [2] benson atlas
Step [1] will create the subject specific freesurfer folder. Now we use this to create the benson atlas
```bash
conda activate b14 # python environment we need
s02_b14atlas.py <sub-id> --SUBJECTS_DIR $SUBJECTS_DIR
```
This will add benson atlas to your freesurfer folder and add the ROIS to your label folder 

---
## Following steps are optional - but may be useful for visualisation
## [3] autoflatten 
Makes cuts to your inflated freesurfer surface so that they can be easily used as flatmaps

```bash
conda activate autoflat
autoflatten $SUBJECTS_DIR/<sub-id>
```
## [4] pycortex 
Takes the freesurfer + autoflatten outputs and makes a subject specific pycortex directory. this can be used for making quick flatmaps. It also automatically imports the freesurfer ROIs on top. 

```bash
conda activate pctx
s04_pycortex.py <sub-id> --fsdir $SUBJECTS_DIR
```