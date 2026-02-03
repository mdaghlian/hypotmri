# pybest - todo 
one way of denoising...
```bash
conda activate pybest02
export SUBJECTS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/freesurfer
PYB_DIR=$SUBJECTS_DIR/../pybest20
FPREP_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/fmriprep
mkdir -p $PYB_DIR
pybest ${FPREP_DIR} \
    --subject hp01 --space fsnative --n-comps 20 \
    --hemi L --out-dir $PYB_DIR --save-all

```