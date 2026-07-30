#!/bin/bash

sub="sub-hp01"
ses="ses-01"
task="task-pRFRE"
run="run-02"
tr=1.0

export SUBJECTS_DIR=~/projects/dp-clean-link/240522NG/hypot/derivatives/freesurfer
BIDS_DIR=~/projects/dp-clean-link/240522NG/hypot/
DERIV_DIR=~/projects/dp-clean-link/240522NG/hypot/derivatives
MC_DIR=${DERIV_DIR}/sf1_mc_sdc/${sub}/${ses}

# find aseg file
aseg_file=${SUBJECTS_DIR}/${sub}/mri/aparc+aseg.mgz

# find sub-*_ses-*_task-*_run-01_bold_desc-mcflirt_motion.par
tMC_FILE=$(find ${MC_DIR} -name "${sub}_${ses}_${task}_${run}_bold_desc-mcflirt_motion.par" | head -n 1)
echo "Motion parameters file: ${tMC_FILE}"

# find mc bold 
tBOLD=$(find ${MC_DIR} -name "${sub}_${ses}_${task}_${run}_bold_desc-preproc_bold.nii.gz" | head -n 1)
echo "Motion corrected BOLD file: ${tBOLD}"

CONFOUND_DIR=${DERIV_DIR}/sf4_confounds
rm -rf $OUT_DIR
OUT_DIR=${CONFOUND_DIR}/${sub}/${ses}
if [ ! -d ${OUT_DIR} ]; then
    mkdir -p ${OUT_DIR}
fi

python ./s04_generate_confounds_nipy.py \
    --bold "${tBOLD}" \
    --mcpar "${tMC_FILE}" \
    --aseg "${aseg_file}" \
    --tr ${tr} \
    --outdir "${OUT_DIR}" \
    --outfile "${sub}_${ses}_${task}_${run}_desc-confounds_timeseries"