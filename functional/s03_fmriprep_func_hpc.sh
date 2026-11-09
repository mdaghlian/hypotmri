#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --sub <sub> [--ses <ses> ...]"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --input-file    input file, placed in BID_DIR/derivatives"
    echo ""
    echo "Optional Arguments:"
    echo "  --ses           Session label (e.g., ses-01). Repeat --ses to process"
    echo "                  multiple sessions in a single fMRIPrep run. Omit"
    echo "                  entirely to auto-discover and process every session"
    echo "                  available for this subject."
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
SESSIONS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSIONS+=("$2"); shift 2 ;;
        --input-file)       INPUT_FILE="$2"; shift 2 ;;
        --no-qsub)          NO_QSUB=true; shift ;;
        --skip-sync)        SKIP_SYNC=true; shift ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session(s) robust, and reject anything that isn't a
# clean alphanumeric BIDS label once 'ses-' is stripped. This catches the
# case where multiple sessions were meant but not space-separated - e.g.
# '--ses 01-02' typed meaning sessions 01 and 02 - which would otherwise
# silently become one bogus session called 'ses-01-02'.
SUBJECT="sub-${SUBJECT#sub-}"
for i in "${!SESSIONS[@]}"; do
    SESSIONS[$i]="ses-${SESSIONS[$i]#ses-}"
    stripped="${SESSIONS[$i]#ses-}"
    if [[ ! "$stripped" =~ ^[a-zA-Z0-9]+$ ]]; then
        echo "Error: invalid session label '${SESSIONS[$i]}' - BIDS session labels are alphanumeric only."
        echo "If you meant multiple sessions, pass them as separate --ses flags, e.g. --ses 01 --ses 02 (not --ses 01-02)."
        exit 1
    fi
done

# --- Validate ---
[[ -z "$BIDS_DIR" ]] && echo "Error: --bids-dir required" && usage
[[ -z "$SUBJECT" ]]  && echo "Error: --sub required"      && usage
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

# --- Auto-discover sessions if none were given ---
# Mirrors s03_fmriprep_func.py's own auto-discovery. We can only resolve
# this here (rather than leaving it entirely to the python side) when
# $BIDS_DIR is something this shell can actually list - true when running
# directly on HPC, or locally when we're about to rsync anyway. The one
# case we can't resolve here is "local machine + --skip-sync", where the
# local machine may not hold a copy of the data at all - in that case we
# leave SESSIONS empty and let the remote python script auto-discover
# once it's running against the real BIDS_DIR.
if [[ ${#SESSIONS[@]} -eq 0 ]] && ! { [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" == true ]]; }; then
    echo "No --ses given — auto-discovering sessions under \$BIDS_DIR/derivatives/${INPUT_FILE}/${SUBJECT} ..."
    mapfile -t SESSIONS < <(
        find "${BIDS_DIR}/derivatives/${INPUT_FILE}/${SUBJECT}" \
            -mindepth 1 -maxdepth 1 -type d -name 'ses-*' \
            -exec basename {} \; 2>/dev/null | sort
    )
    if [[ ${#SESSIONS[@]} -eq 0 ]]; then
        echo "Error: no session directories found under ${BIDS_DIR}/derivatives/${INPUT_FILE}/${SUBJECT}"
        exit 1
    fi
    echo "Auto-discovered session(s): ${SESSIONS[*]}"
fi


# --- Rsync input-file to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing input file to cluster..."
    echo " ASSUMING FMRIPREP & FREESURFER outputs are already present on cluster (from previous steps)"
    for SES in "${SESSIONS[@]}"; do
        bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
            --bids-dir "$BIDS_DIR" \
            --sub      "$SUBJECT" \
            --ses      "$SES" \
            --deriv "${INPUT_FILE}"
    done
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
if [[ ${#SESSIONS[@]} -gt 0 ]]; then
    SESSIONS_STR="${SESSIONS[*]}"
else
    SESSIONS_STR="(none given - auto-discover all on execution host)"
fi
echo "  Subject:           $SUBJECT"
echo "  Sessions:          $SESSIONS_STR"
echo "  No-qsub mode:      $NO_QSUB"
echo "-------------------------------------------------------"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"

# Build a quoted --ses arg per session so all sessions are staged and
# processed in a single fMRIPrep run. When SESSIONS is empty (only
# possible in the local + --skip-sync case above), omit --ses entirely
# so the python script auto-discovers sessions itself.
if [[ ${#SESSIONS[@]} -gt 0 ]]; then
    JOB_TAG=$(IFS=- ; echo "${SESSIONS[*]}")
    SES_ARGS=""
    for SES in "${SESSIONS[@]}"; do
        SES_ARGS="${SES_ARGS} '${SES}'"
    done
    SES_FLAG="--ses ${SES_ARGS}"
else
    JOB_TAG="allsessions"
    SES_FLAG=""
fi
JOB_NAME="fprep_func_${SUBJECT}_${JOB_TAG}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

RUNNER_SCRIPT="~/pipeline/functional/s03_fmriprep_func.py \
    --bids-dir '${SUBMIT_BIDS_DIR}' \
    --sub      '${SUBJECT}' \
    ${SES_FLAG} \
    --input-file '${INPUT_FILE}'"
echo $RUNNER_SCRIPT
# Make sure output dir exists
[[ ! -d "${BIDS_DIR}/derivatives/fmriprep" ]] && mkdir -p "${BIDS_DIR}/derivatives/fmriprep"

# --- Submit via qsub ---
echo "-------------------------------------------------------"
echo "Submitting fmriprep functional job"
echo "  Subject:  $SUBJECT"
echo "  Sessions: $SESSIONS_STR"
echo "  Logs:     ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
echo "-------------------------------------------------------"
QSUB_CMD="source ~/.bash_profile; \
    source set_project.sh ${PROJ_NAME}; \
    mkdir -p '${REMOTE_LOG_DIR}'; \
    conda activate preproc; \
    source set_remote_paths.sh;\
    qsub -V \
            -N  '${JOB_NAME}' \
            -o  '${LOG_OUT}' \
            -e  '${LOG_ERR}' \
            -l  h_rt=8:00:00 \
            -l  mem=32G \
            -l  tmpfs=200G \
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
