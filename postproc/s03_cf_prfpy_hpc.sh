#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --input-file <path> --output-file <path> --sub <sub> --ses <ses> --task <task> --project <project> --roi-src <roi> --roi-target <roi>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --input-file    Input derivatives directory (surface time series)"
    echo "  --output-file   Output derivatives directory for CF fits"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo "  --task          Task label (e.g., pRFLE)"
    echo "  --project       Project name for settings/dm file lookup"
    echo "  --roi-src       Source ROI label (e.g., b14_V1.)"
    echo "  --roi-target    Target ROI label (e.g., b14_ALL)"
    echo ""
    echo "Optional Arguments:"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --overwrite     Space-separated step names to overwrite (psc_concat gdist grid_fit)"
    echo "  --overwrite-all Overwrite all steps"
    echo "  --skip          Space-separated step names to skip"
    echo "  --help          Display this help message"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync input file to cluster (if running from local)
# [2] qsub s03_cf_prfpy.py
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- --- ---

# --- Parse Arguments ---
SKIP_SYNC=false
OVERWRITE_STEPS=""
OVERWRITE_ALL=false
SKIP_STEPS=""
ROI_SRC="b14_V1."
ROI_TARGET="b14_ALL"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2";       shift 2 ;;
        --input-file)       INPUT_FILE="$2";     shift 2 ;;
        --output-file)      OUTPUT_FILE="$2";    shift 2 ;;
        --sub)              SUBJECT="$2";        shift 2 ;;
        --ses)              SESSION="$2";        shift 2 ;;
        --task)             TASK="$2";           shift 2 ;;
        --project)          PROJECT="$2";        shift 2 ;;
        --roi-src)          ROI_SRC="$2";        shift 2 ;;
        --roi-target)       ROI_TARGET="$2";     shift 2 ;;
        --skip-sync)        SKIP_SYNC=true;      shift ;;
        --overwrite)        OVERWRITE_STEPS="$2"; shift 2 ;;
        --overwrite-all)    OVERWRITE_ALL=true;  shift ;;
        --skip)             SKIP_STEPS="$2";     shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session labels robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]     && echo "Error: --bids-dir required"     && usage
[[ -z "$INPUT_FILE" ]]   && echo "Error: --input-file required"   && usage
[[ -z "$OUTPUT_FILE" ]]  && echo "Error: --output-file required"  && usage
[[ -z "$SUBJECT" ]]      && echo "Error: --sub required"          && usage
[[ -z "$SESSION" ]]      && echo "Error: --ses required"          && usage
[[ -z "$TASK" ]]         && echo "Error: --task required"         && usage
[[ -z "$PROJECT" ]]      && echo "Error: --project required"      && usage

# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    [[ -z "$REMOTE_BIDS_DIR" ]] && echo "Error: \$REMOTE_BIDS_DIR not set in environment" && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi

# --- Rsync input data to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing ${INPUT_FILE}"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --deriv    "$INPUT_FILE"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running CONNECTIVE FIELD FITTING"
echo "-------------------------------------------------------"
if [[ "${PC_LOCATION}" == "local" ]]; then
    echo "  Running from:         local"
    echo "  BIDS DIR (local):     $BIDS_DIR"
    echo "  BIDS DIR (remote):    $REMOTE_BIDS_DIR"
else
    echo "  Running from:         HPC (direct qsub)"
    echo "  BIDS DIR:             $BIDS_DIR"
fi
echo "  Subject:              $SUBJECT"
echo "  Session:              $SESSION"
echo "  Task:                 $TASK"
echo "  Project:              $PROJECT"
echo "  ROI source:           $ROI_SRC"
echo "  ROI target:           $ROI_TARGET"
echo "  Input:                $INPUT_FILE"
echo "  Output:               $OUTPUT_FILE"
echo "-------------------------------------------------------"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
JOB_NAME="cf_prfpy_${SUBJECT}_${SESSION}_task-${TASK}_roisrc-${ROI_SRC}_roitarget-${ROI_TARGET}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

# Build optional flags
OVERWRITE_FLAG=""
[[ -n "$OVERWRITE_STEPS" ]] && OVERWRITE_FLAG="--overwrite ${OVERWRITE_STEPS}"
[[ "$OVERWRITE_ALL" == true ]] && OVERWRITE_FLAG="--overwrite-all"

SKIP_FLAG=""
[[ -n "$SKIP_STEPS" ]] && SKIP_FLAG="--skip ${SKIP_STEPS}"

RUNNER_SCRIPT="~/pipeline/postproc/s03_cf_prfpy.py \
    --bids-dir    '${SUBMIT_BIDS_DIR}' \
    --input-file  '${INPUT_FILE}' \
    --output-file '${OUTPUT_FILE}' \
    --sub         '${SUBJECT}' \
    --ses         '${SESSION}' \
    --task        '${TASK}' \
    --project     '${PROJECT}' \
    --roi-src     '${ROI_SRC}' \
    --roi-target  '${ROI_TARGET}' \
    ${OVERWRITE_FLAG} \
    ${SKIP_FLAG}"

echo "-------------------------------------------------------"
echo "Submitting CF PRFPY job"
echo "  Subject:    $SUBJECT"
echo "  Session:    $SESSION"
echo "  Task:       $TASK"
echo "  ROI src:    $ROI_SRC"
echo "  ROI target: $ROI_TARGET"
echo "  Logs:       ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
echo "-------------------------------------------------------"

QSUB_CMD="source ~/.bash_profile; \
    source set_project.sh ${PROJ_NAME}; \
    conda activate prf; \
    mkdir -p '${REMOTE_LOG_DIR}'; \
    qsub -V \
        -N  '${JOB_NAME}' \
        -o  '${LOG_OUT}' \
        -e  '${LOG_ERR}' \
        -l  h_rt=24:00:00 \
        -l  mem=32G \
        -pe smp 8 \
        -j  n \
        ${RUNNER_SCRIPT}"

if [[ "${PC_LOCATION}" == "local" ]]; then
    JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
else
    JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
fi
echo "Submitted job ID: ${JOB_ID}"
