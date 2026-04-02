#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --sub <sub> --ses <ses> --input-file <input-file>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path BIDS directory "
    echo "  --input-file    Name of derivative file containing preprocessed BOLD"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}
# --- SCRTIPT OVERVIEW ---
# [1] Look for FPREP_BIDS inside BIDS_DIR/derivatives
# -- Why? We don't want to run fmriprep on the actual "raw" data
# -- but rather on a subset of the data we have preprocessed. To 
# -- do this we put only what we want inside "FPREPBIDS"
# -- The fMRIPREP+Freesurfer outputs are placed in the usual place
# -- (BIDS_DIR/derivatives)
# [2] Copy the preprocessed BOLD files from input-dir to FPREP_BIDS
# [3] Run fmriprep
# --- --- --- --- 

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSION="$2"; shift 2 ;;
        --input-file)       INPUT_FILE="$2"; shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done
# -> make subject & session robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

INPUT_DIR="${BIDS_DIR}/derivatives/${INPUT_FILE}"
SUBJECT_INPUT_DIR="${INPUT_DIR}/${SUBJECT}/${SESSION}"
[[ ! -d "${SUBJECT_INPUT_DIR}" ]] && echo "Error: Input file ${SUBJECT_INPUT_DIR} not found" && exit 1
# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - to get the confounds "
echo "-------------------------------------------------------"
echo " Input:     $INPUT_DIR"
echo " Output:    $BIDS_DIR"
echo " Subject:   $SUBJECT"
echo " Session:   $SESSION"
echo "-------------------------------------------------------"

# Construct paths
FPREP_BIDS_DIR=$BIDS_DIR/derivatives/FPREP_BIDS
if [[ ! -d "${FPREP_BIDS_DIR}" ]]; then
    mkdir -p ${FPREP_BIDS_DIR}
fi 
FPREP_BIDS_DIR_WF=${BIDS_DIR}/derivatives/FPREP_BIDS_WF
if [[ ! -d "${FPREP_BIDS_DIR_WF}" ]]; then
    mkdir -p $FPREP_BIDS_DIR_WF
fi 

# Find all the BOLD runs
# sub-XX_ses-XX_*space-fsT1*.nii.gz
BOLD_FILES=($(find "${INPUT_DIR}" -name "${SUBJECT}_${SESSION}*space-fsT1_*.nii*" | sort))    
if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}_task-${TASK}"
    exit 1
else
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
fi
echo "BOLD files:"
for f in "${BOLD_FILES[@]}"; do
    # print basename
    echo "  - $(basename "$f")"
done
# Copy each run to FPREP_BIDS with standardized naming:
FPREP_FUNC_DIR="${FPREP_BIDS_DIR}/${SUBJECT}/${SESSION}/func"
if [[ ! -d "${FPREP_FUNC_DIR}" ]]; then
    mkdir -p "${FPREP_FUNC_DIR}"
fi

run_counter=0
for BOLD_FILE in "${BOLD_FILES[@]}"; do
    # Get clean run name as: 
    # -> sub-XX_ses-XX_task-XX_run-XX_bold.nii.gz
    BOLD_BASENAME=$(basename "$BOLD_FILE")
    # Extract task and run labels if present
    task_part=""
    if [[ "$BOLD_BASENAME" =~ task-([a-zA-Z0-9]+) ]]; then
        task_label="task-${BASH_REMATCH[1]}"
        task_part="_${task_label}"
    fi
    run_part=""
    if [[ "$BOLD_BASENAME" =~ run-([0-9]+) ]]; then
        run_part="_run-${BASH_REMATCH[1]}"
    fi
    clean_run_name="${SUBJECT}_${SESSION}${task_part}${run_part}_bold.nii.gz"
    echo "Processing run: ${clean_run_name}"
    FPREP_BOLD="${FPREP_FUNC_DIR}/${clean_run_name}"
    FPREP_JSON="${FPREP_FUNC_DIR}/${clean_run_name%.nii.gz}.json"
    BOLD_TR=$(conda run -n preproc fslval "$BOLD_FILE" pixdim4)
cat <<EOF > "$FPREP_JSON"
{
  "RepetitionTime": $BOLD_TR,
  "TaskName": "${TASK_LABEL}"
}
EOF
    cp $BOLD_FILE $FPREP_BOLD
    
done

echo ""
echo "=========================================="
echo "Copied it all over - now for fmriprep"
echo "=========================================="
# MD note higher fd-spike 0.9
  
[[ ! -d "${BIDS_DIR}/derivatives/fmriprep" ]] && mkdir -p "${BIDS_DIR}/derivatives/fmriprep"
if [[ "$CONTAINER_TYPE" == "docker" ]]; then
    docker run --rm \
      -v $FPREP_BIDS_DIR:/data:ro \
      -v $BIDS_DIR/derivatives/fmriprep:/out \
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
        --omp-nthreads 8 --nprocs 8 \
        --ignore fieldmaps slicetiming \
        --output-spaces func \

elif [[ "$CONTAINER_TYPE" == "apptainer" || "$CONTAINER_TYPE" == "singularity" ]]; then
    ${CONTAINER_TYPE} run \
      --cleanenv \
      -B $FPREP_BIDS_DIR:/data \
      -B $BIDS_DIR/derivatives/fmriprep:/out \
      -B $FPREP_BIDS_DIR_WF:/work \
      -B $SUBJECTS_DIR:/fsdir \
      -B $PIPELINE_DIR/config/license.txt:/license.txt \
      ${SIF_DIR}/${FPREP_SIF} \
        /data /out participant \
        --participant-label $SUBJECT \
        --skip_bids_validation \
        --fs-subjects-dir /fsdir \
        --fs-license-file /license.txt \
        --work-dir /work \
        --omp-nthreads 8 --nprocs 8 \
        --ignore fieldmaps slicetiming \
        --output-spaces func \

else
    echo "Invalid CONTAINER_TYPE: $CONTAINER_TYPE"
    exit 1
fi