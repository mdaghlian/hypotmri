#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --output-file <string> --sub <sub> --ses <ses>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --output-file   output file, placed in BID_DIR/derivatives"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --task          Task label (default: empty, matches all tasks)"
    echo "  --no-qsub       Run the pipeline script directly (no job submission)"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --help          Display this help message"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync raw func + fmap data to cluster (if running from local)
# [2] qsub s01_sdc_AFNI.py
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- --- ---

# --- Parse Arguments ---
NO_QSUB=false
SKIP_SYNC=false
TASK=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2";   shift 2 ;;
        --output-file)      OUTPUT_FILE="$2"; shift 2 ;;
        --sub)              SUBJECT="$2";    shift 2 ;;
        --ses)              SESSION="$2";    shift 2 ;;
        --task)             TASK="$2";       shift 2 ;;
        --no-qsub)          NO_QSUB=true;   shift ;;
        --skip-sync)        SKIP_SYNC=true; shift ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session labels robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]    && echo "Error: --bids-dir required"    && usage
[[ -z "$OUTPUT_FILE" ]]  && echo "Error: --output-file required"  && usage
[[ -z "$SUBJECT" ]]     && echo "Error: --sub required"         && usage
[[ -z "$SESSION" ]]     && echo "Error: --ses required"         && usage

# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    [[ -z "$REMOTE_BIDS_DIR" ]]   && echo "Error: \$REMOTE_BIDS_DIR not set in environment"   && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi
# --- Rsync func + fmap data to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing func/fmap data to cluster + freesurfer"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --raw --deriv freesurfer
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running SDC (AFNI method)"
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
echo "  Task:                 ${TASK:-<all>}"
echo "  No-qsub mode:         $NO_QSUB"
echo "-------------------------------------------------------"

# --- Build task suffix for job naming (safe even if TASK is empty) ---
TASK_SUFFIX="${TASK:+_task-${TASK}}"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
JOB_NAME="sdc_afni_${SUBJECT}_${SESSION}${TASK_SUFFIX}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

# Build optional --task flag (omit entirely if TASK is empty)
TASK_ARG="${TASK:+--task '${TASK}'}"

RUNNER_SCRIPT="~/pipeline/functional/s01_sdc_AFNI.py \
    --bids-dir    '${SUBMIT_BIDS_DIR}' \
    --output-file '${OUTPUT_FILE}' \
    --sub         '${SUBJECT}' \
    --ses         '${SESSION}' \
    --task        '${TASK}' \
    --afni-docker '${AFNI_SIF}'"


if [[ "$NO_QSUB" == true ]]; then
    echo "-------------------------------------------------------"
    echo "Running SDC AFNI directly (no qsub)"
    echo "  Subject:  $SUBJECT"
    echo "  Session:  $SESSION"
    echo "  Task:     ${TASK:-<all>}"
    echo "-------------------------------------------------------"
    if [[ "${PC_LOCATION}" == "local" ]]; then
        ssh "$REMOTE_HOST" "$RUNNER_SCRIPT"
    else
        eval "$RUNNER_SCRIPT"
    fi
else
    echo "-------------------------------------------------------"
    echo "Submitting SDC AFNI job"
    echo "  Subject:  $SUBJECT"
    echo "  Session:  $SESSION"
    echo "  Task:     ${TASK:-<all>}"
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
            -l  h_rt=1:00:00 \
            -l  mem=16G \
            -pe smp 1 \
            -j  n \
            ${RUNNER_SCRIPT}"
    echo "$QSUB_CMD"
    
    if [[ "${PC_LOCATION}" == "local" ]]; then
        JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
    else
        JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
    fi
    echo "Submitted job ID: ${JOB_ID}"
fi