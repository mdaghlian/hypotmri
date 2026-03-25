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
    echo ""
    echo "Optional Arguments:"
    echo "  --no-qsub       Run the pipeline script directly (no job submission)"
    echo "  --help          Display this help message"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync raw anat to cluster (if running from local, skipped if on HPC)
# [2] qsub s01_fmriprep_anat_only.sh
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- ---

# --- Parse Arguments ---
NO_QSUB=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSION="$2"; shift 2 ;;
        --no-qsub)          NO_QSUB=true; shift ;;
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


# --- Rsync anat to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    echo "Rsyncing anat data to cluster..."
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --raw-anat
    # Also make sure that the scripts are up to date
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - anatomy only"
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
JOB_NAME="fprep_anat_${SUBJECT}_${SESSION}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

RUNNER_SCRIPT="~/pipeline/anatomical/s01_fmriprep_anat_only.sh \
    --bids-dir '${SUBMIT_BIDS_DIR}' \
    --sub      '${SUBJECT}' \
    --ses      '${SESSION}'"
# Make sure output dir exists 
[[ ! -d "${BIDS_DIR}/derivatives/fmriprep" ]] && mkdir -p "${BIDS_DIR}/derivatives/fmriprep"
if [[ "$NO_QSUB" == true ]]; then
    # --- Run directly (no job scheduler) ---
    echo "-------------------------------------------------------"
    echo "Running fmriprep anat-only directly (no qsub)"
    echo "  Subject:  $SUBJECT"
    echo "  Session:  $SESSION"
    echo "-------------------------------------------------------"
    DIRECT_CMD="SUBJECT='${SUBJECT}' SESSION='${SESSION}' CONTAINER_TYPE='apptainer' \
        bash ${RUNNER_SCRIPT}"
    if [[ "${PC_LOCATION}" == "local" ]]; then
        ssh "$REMOTE_HOST" "$DIRECT_CMD"
    else
        eval "$DIRECT_CMD"
    fi
else
    # --- Submit via qsub ---
    echo "-------------------------------------------------------"
    echo "Submitting fmriprep anat-only job"
    echo "  Subject:  $SUBJECT"
    echo "  Session:  $SESSION"
    echo "  Logs:     ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
    echo "-------------------------------------------------------"
    QSUB_CMD="mkdir -p '${REMOTE_LOG_DIR}' && qsub -V \
        -N  '${JOB_NAME}' \
        -o  '${LOG_OUT}' \
        -e  '${LOG_ERR}' \
        -l  h_rt=12:00:00 \
        -l  mem=8G \
        -pe smp 1 \
        -j  n \
        -v  SUBJECT='${SUBJECT}',SESSION='${SESSION}',CONTAINER_TYPE='apptainer' \
        ${RUNNER_SCRIPT}"
    # -l  tmpfs=50G \ ? 
    echo "$QSUB_CMD"
    if [[ "${PC_LOCATION}" == "local" ]]; then
        JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
    else
        JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
    fi
    echo "Submitted job ID: ${JOB_ID}"
fi