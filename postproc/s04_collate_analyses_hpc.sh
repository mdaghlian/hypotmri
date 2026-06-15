#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --prf-file <path> --cf-file <path> --output-file <path> --sub <sub>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --prf-file      Derivatives directory containing pRF fit CSVs"
    echo "  --cf-file       Derivatives directory containing CF fit CSVs"
    echo "  --output-file   Output derivatives directory for collated CSV"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --overwrite     Overwrite existing combined csv"
    echo "  --overwrite-all Overwrite existing combined csv (same as --overwrite)"
    echo "  --help          Display this help message"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync prf-file and cf-file derivatives to cluster (if running from local)
# [2] qsub s04_collate_analyses.py
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- --- ---

# --- Parse Arguments ---
SKIP_SYNC=false
OVERWRITE_ALL=false
OVERWRITE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2";       shift 2 ;;
        --prf-file)         PRF_FILE="$2";       shift 2 ;;
        --cf-file)          CF_FILE="$2";        shift 2 ;;
        --output-file)      OUTPUT_FILE="$2";    shift 2 ;;
        --sub)              SUBJECT="$2";        shift 2 ;;
        --skip-sync)        SKIP_SYNC=true;      shift ;;
        --overwrite)        OVERWRITE=true;      shift ;;
        --overwrite-all)    OVERWRITE_ALL=true;  shift ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject label robust
SUBJECT="sub-${SUBJECT#sub-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]     && echo "Error: --bids-dir required"     && usage
[[ -z "$PRF_FILE" ]]     && echo "Error: --prf-file required"     && usage
[[ -z "$CF_FILE" ]]      && echo "Error: --cf-file required"      && usage
[[ -z "$OUTPUT_FILE" ]]  && echo "Error: --output-file required"  && usage
[[ -z "$SUBJECT" ]]      && echo "Error: --sub required"          && usage

# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    [[ -z "$REMOTE_BIDS_DIR" ]] && echo "Error: \$REMOTE_BIDS_DIR not set in environment" && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi

# --- Rsync derivatives to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing ${PRF_FILE}"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --deriv    "$PRF_FILE"
    echo "Rsyncing ${CF_FILE}"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --deriv    "$CF_FILE"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running COLLATE ANALYSES"
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
echo "  PRF file:             $PRF_FILE"
echo "  CF file:              $CF_FILE"
echo "  Output:               $OUTPUT_FILE"
echo "-------------------------------------------------------"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
JOB_NAME="collate_${SUBJECT}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

# Build optional flags
OVERWRITE_FLAG=""
[[ "$OVERWRITE" == true ]] && OVERWRITE_FLAG="--overwrite"
[[ "$OVERWRITE_ALL" == true ]] && OVERWRITE_FLAG="--overwrite-all"

RUNNER_SCRIPT="~/pipeline/postproc/s04_collate_analyses.py \
    --bids-dir    '${SUBMIT_BIDS_DIR}' \
    --prf-file    '${PRF_FILE}' \
    --cf-file     '${CF_FILE}' \
    --output-file '${OUTPUT_FILE}' \
    --sub         '${SUBJECT}' \
    ${OVERWRITE_FLAG}"

echo "-------------------------------------------------------"
echo "Submitting COLLATE ANALYSES job"
echo "  Subject:  $SUBJECT"
echo "  Logs:     ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
echo "-------------------------------------------------------"

QSUB_CMD="source ~/.bash_profile; \
    source set_project.sh ${PROJ_NAME}; \
    conda activate prf; \
    mkdir -p '${REMOTE_LOG_DIR}'; \
    qsub -V \
        -N  '${JOB_NAME}' \
        -o  '${LOG_OUT}' \
        -e  '${LOG_ERR}' \
        -l  h_rt=1:00:00 \
        -l  mem=8G \
        -pe smp 1 \
        -j  n \
        ${RUNNER_SCRIPT}"

if [[ "${PC_LOCATION}" == "local" ]]; then
    JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
else
    JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
fi
echo "Submitted job ID: ${JOB_ID}"
