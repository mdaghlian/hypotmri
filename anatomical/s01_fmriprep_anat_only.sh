#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --BIDS_DIR <path> --sub <sub> --ses <ses>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path BIDS directory "
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}
# --- SCRTIPT OVERVIEW ---
# [1] Create FPREPBIDS -> inside BIDS_DIR/derivatives
# -- Why? We don't want to run fmriprep on the actual "raw" data
# -- but rather on a subset of the data we have preprocessed. To 
# -- do this we put only what we want inside "FPREPBIDS"
# -- The fMRIPREP+Freesurfer outputs are placed in the usual place
# -- (BIDS_DIR/derivatives)
# [2] Copy the anatomical (T1w) from BIDS_DIR/SUBJECTS/SESSION/anat
# [3] Run fmriprep
# --- --- --- --- 

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSION="$2"; shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - for anatomy "
echo "-------------------------------------------------------"
echo " BIDS DIR:    $BIDS_DIR"
echo " Subject:   $SUBJECT"
echo " Session:   $SESSION"
echo "-------------------------------------------------------"

# [1] Create FPREP BIDS
FPREP_BIDS_DIR=$BIDS_DIR/derivatives/FPREP_BIDS
if [[ ! -f "${FPREP_BIDS_DIR}" ]]; then
    mkdir -p ${FPREP_BIDS_DIR}
fi 
FPREP_BIDS_DIR_WF=${BIDS_DIR}/derivatives/FPREP_BIDS_WF
if [[ ! -f "${FPREP_BIDS_DIR_WF}" ]]; then
    mkdir -p $FPREP_BIDS_DIR_WF
fi 
# -> Create bids json if it doesn't exist
BIDS_JSON="${FPREP_BIDS_DIR}/dataset_description.json"
if [[ ! -f "${BIDS_JSON}" ]]; then
    printf "{\"Name\": \"Example dataset\", \"BIDSVersion\": \"1.0.2\"}" >> "$BIDS_JSON"
fi

# -> Create freesurfer output, if it doesn't exist
# Note this is inside the "true" BIDS_DIR 
SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
if [[ ! -f "${SUBJECTS_DIR}" ]]; then
    mkdir -p "${SUBJECTS_DIR}"
fi

FPREP_SES="${FPREP_BIDS_DIR}/${SUBJECT}/${SESSION}"
if [[ -e "${FPREP_SES}" ]]; then
    rm -rf ${FPREP_SES}
fi
mkdir -p "${FPREP_SES}"

echo "Copying anatomy" 
ANAT_SRC="${BIDS_DIR}/${SUBJECT}/ses-01/anat"
cp -r ${ANAT_SRC} ${FPREP_SES}/

if [[ "$CONTAINER_TYPE" == "docker" ]]; then
    docker run --rm \
      -v $FPREP_BIDS_DIR:/data:ro \
      -v $BIDS_DIR/derivatives:/out \
      -v $FPREP_BIDS_DIR_WF:/work \
      -v $SUBJECTS_DIR:/fsdir \
      -v $PIPELINE_DIR/config/license.txt:/license.txt \
      $FPREP_IMAGE \
        /data /out participant \
        --participant-label $SUBJECT \
        --skip_bids_validation \
        --fs-subjects-dir /fsdir \
        --fs-license-file /license.txt \
        --work-dir /work \
        --anat-only \
        --omp-nthreads 8 --nprocs 8 --output-layout legacy 

elif [[ "$CONTAINER_TYPE" == "apptainer" || "$CONTAINER_TYPE" == "singularity" ]]; then
    ${CONTAINER_TYPE} run \
      --cleanenv \
      -B $FPREP_BIDS_DIR:/data \
      -B $BIDS_DIR/derivatives:/out \
      -B $FPREP_BIDS_DIR_WF:/work \
      -B $SUBJECTS_DIR:/fsdir \
      -B $PIPELINE_DIR/config/license.txt:/license.txt \
      $FPREP_SIF \
        /data /out participant \
        --participant-label $SUBJECT \
        --skip_bids_validation \
        --fs-subjects-dir /fsdir \
        --fs-license-file /license.txt \
        --work-dir /work \
        --anat-only \
        --omp-nthreads 8 --nprocs 8
else
    echo "Invalid CONTAINER_TYPE: $CONTAINER_TYPE"
    exit 1
fi
# Create a symlink between subject dir and the annoying way that fmriprep does 
# freesurfer naming. See:
# https://neurostars.org/t/automatic-freesurfer-subject-name-includes-session-tag/35135/2
# https://github.com/nipreps/fmriprep/pull/3588
SUB_FS=$SUBJECTS_DIR/$SUBJECT
SUB_FS_FPREP=$SUBJECTS_DIR/${SUBJECT}_${SESSION}
if [[ ! -d "${SUB_FS}" ]]; then
    ln -s $SUB_FS_FPREP $SUB_FS
fi