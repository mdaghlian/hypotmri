#!/bin/bash

sub="sub-hp01"
ses="ses-01"
task="task-pRFRE"
run="run-01"
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

OUT_DIR=${DERIV_DIR}/sf4_confounds/${sub}/${ses}
if [ ! -d ${OUT_DIR} ]; then
    mkdir -p ${OUT_DIR}
fi
OUT_FILE=${OUT_DIR}/${sub}_${ses}_${task}_${run}_desc-confounds_timeseries.tsv
echo "Output confounds file: ${OUT_FILE}"

python ./s04_generate_confounds.py \
    --bold "${tBOLD}" \
    --aseg "${aseg_file}" \
    --motion "${tMC_FILE}" \
    --out "${OUT_FILE}" \
    --acompcor 5 \
    --tcompcor 6 \
    --tcompcor-percent 2.0 \
    --erosion 1 \
    --tr ${tr} \
    --highpass 128.0 \
    --fd-thresh 0.5 \
    --dvars-thresh 1.5
