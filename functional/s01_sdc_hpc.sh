#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --output-file <string> --sub <sub> --ses <ses> [options] [-- extra args]"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --output-file   output file, placed in BID_DIR/derivatives"
    echo "  --sub           Subject label (e.g., sub-01 or 01)"
    echo "  --ses           Session label (e.g., ses-01 or 01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --task          Task label (default: empty, matches all tasks)"
    echo "  --no-qsub       Run the pipeline script directly (no job submission)"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --help          Display this help message"
    echo ""
    echo "Any arguments after '--' are forwarded directly to s01_sdc_AFNI.py"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync raw func + fmap data to cluster (if running from local)
# [2] qsub s01_sdc_AFNI.py (or run directly)
# --- --- --- --- ---

# --- Defaults ---
NO_QSUB=false
SKIP_SYNC=false
TASK=""
EXTRA_ARGS=()

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2";   shift 2 ;;
        --output-file)      OUTPUT_FILE="$2"; shift 2 ;;
        --sub)              SUBJECT="$2";    shift 2 ;;
        --ses)              SESSION="$2";    shift 2 ;;
        --task)             TASK="$2";       shift 2 ;;
        --no-qsub)          NO_QSUB=true;     shift ;;
        --skip-sync)        SKIP_SYNC=true;   shift ;;
        --help)             usage ;;
        --)                 shift; EXTRA_ARGS+=("$@"); break ;;
        *)                  EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# --- Normalize subject/session labels ---
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]     && echo "Error: --bids-dir required"     && usage
[[ -z "$OUTPUT_FILE" ]]  && echo "Error: --output-file required"  && usage
[[ -z "$SUBJECT" ]]      && echo "Error: --sub required"          && usage
[[ -z "$SESSION" ]]      && echo "Error: --ses required"          && usage

# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    [[ -z "$REMOTE_BIDS_DIR" ]] && echo "Error: \$REMOTE_BIDS_DIR not set" && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi

# --- Rsync (local only) ---
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
echo "  Subject:              $SUBJECT"
echo "  Session:              $SESSION"
echo "  Task:                 ${TASK:-<all>}"
echo "  No-qsub mode:         $NO_QSUB"
echo "  Forwarded args:       ${EXTRA_ARGS[*]:-<none>}"
echo "-------------------------------------------------------"
exit 1
# --- Build job metadata ---
TASK_SUFFIX="${TASK:+_task-${TASK}}"
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
JOB_NAME="sdc_afni_${SUBJECT}_${SESSION}${TASK_SUFFIX}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

# --- Runner script ---
RUNNER_SCRIPT="~/pipeline/functional/s01_sdc_AFNI.py \
    --bids-dir    '${SUBMIT_BIDS_DIR}' \
    --output-file '${OUTPUT_FILE}' \
    --sub         '${SUBJECT}' \
    --ses         '${SESSION}' \
    ${TASK:+--task '${TASK}'} \
    --afni-docker '${AFNI_SIF}' \
    ${EXTRA_ARGS[*]}"

# --- Run or submit ---
if [[ "$NO_QSUB" == true ]]; then
    echo "Running SDC AFNI directly (no qsub)"
    if [[ "${PC_LOCATION}" == "local" ]]; then
        ssh "$REMOTE_HOST" "$RUNNER_SCRIPT"
    else
        eval "$RUNNER_SCRIPT"
    fi
else
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