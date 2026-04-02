#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --sub <sub> --ses <ses>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo "  --input-file    input file, placed in BID_DIR/derivatives"
    echo ""
    echo "Optional Arguments:"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --no-qsub       Run the pipeline script directly (no job submission)"
    echo "  --help          Display this help message"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync input-file to cluster (if running from local, skipped if on HPC)
# This has the preprocessed BOLD files, which we want to use as input to fmriprep, to get the confounds
# [2] qsub s03_fmriprep_func.sh
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- ---

# --- Parse Arguments ---
NO_QSUB=false
SKIP_SYNC=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSION="$2"; shift 2 ;;
        --input-file)       INPUT_FILE="$2"; shift 2 ;;
        --no-qsub)          NO_QSUB=true; shift ;;
        --skip-sync)        SKIP_SYNC=true; shift ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]] && echo "Error: --bids-dir required" && usage
[[ -z "$SUBJECT" ]]  && echo "Error: --sub required"      && usage
[[ -z "$SESSION" ]]  && echo "Error: --ses required"      && usage
[[ -z "$INPUT_FILE" ]] && echo "Error: --input-file required" && usage
# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    # Local: resolve remote host and path from REMOTE_BIDS_DIR env var
    # Expected format: ucl-work:/path/to/bids
    [[ -z "$REMOTE_BIDS_DIR" ]] && echo "Error: \$REMOTE_BIDS_DIR not set in environment" && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    # On the cluster: BIDS_DIR is already the real path (from .bash_profile)
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi


# --- Rsync input-file to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing input file to cluster..."
    echo " ASSUMING FMRIPREP & FREESURFER outputs are already present on cluster (from previous steps)"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --deriv "${INPUT_FILE}"
    # Also make sure that the scripts are up to date
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - functional"
echo "-------------------------------------------------------"
if [[ "${PC_LOCATION}" == "local" ]]; then
    echo "  Running from:      local"
    echo "  BIDS DIR (local):  $BIDS_DIR"
    echo "  BIDS DIR (remote): $REMOTE_BIDS_DIR"
else
    echo "  Running from:      HPC (direct qsub)"
    echo "  BIDS DIR:          $BIDS_DIR"
fi
echo "  Subject:           $SUBJECT"
echo "  Session:           $SESSION"
echo "  No-qsub mode:      $NO_QSUB"
echo "-------------------------------------------------------"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
JOB_NAME="fprep_func_${SUBJECT}_${SESSION}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

RUNNER_SCRIPT="~/pipeline/functional/s03_fmriprep_func.sh \
    --bids-dir '${SUBMIT_BIDS_DIR}' \
    --sub      '${SUBJECT}' \
    --ses      '${SESSION}' \
    --input-file '${INPUT_FILE}'"
echo $RUNNER_SCRIPT
# Make sure output dir exists 
[[ ! -d "${BIDS_DIR}/derivatives/fmriprep" ]] && mkdir -p "${BIDS_DIR}/derivatives/fmriprep"

# --- Submit via qsub ---
echo "-------------------------------------------------------"
echo "Submitting fmriprep functional job"
echo "  Subject:  $SUBJECT"
echo "  Session:  $SESSION"
echo "  Logs:     ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
echo "-------------------------------------------------------"
QSUB_CMD="source ~/.bash_profile; \
    source set_project.sh ${PROJ_NAME}; \
    mkdir -p '${REMOTE_LOG_DIR}'; \
    conda activate preproc; \
    qsub -V \
        -N  '${JOB_NAME}' \
        -o  '${LOG_OUT}' \
        -e  '${LOG_ERR}' \
        -l  h_rt=10:00:00 \
        -l  mem=2G \
        -pe smp 4 \
        -j  n \
        ${RUNNER_SCRIPT}"
echo "$QSUB_CMD"

if [[ "${PC_LOCATION}" == "local" ]]; then
    JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
else
    JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
fi
echo "Submitted job ID: ${JOB_ID}"
