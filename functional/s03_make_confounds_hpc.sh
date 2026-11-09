#!/bin/bash
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --coreg-file <path> --output-file <path> --sub <sub> --ses <ses>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path to local BIDS directory"
    echo "  --coreg-file    s02_coreg.py output, placed in BIDS_DIR/derivatives"
    echo "  --output-file   output file, placed in BIDS_DIR/derivatives"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --anat-file     s01_fmriprep_anat_only.sh output (default: fmriprep)"
    echo "  --skip-sync     Skip rsync step (assumes data is already on cluster)"
    echo "  --help          Display this help message"
    echo "Any arguments after '--' are forwarded directly to s03_make_confounds.py"
    echo "(e.g. --skip-vols, --variance-threshold, --overwrite, --skip)"
    exit 1
}

# --- SCRIPT OVERVIEW ---
# [1] Rsync coreg-file + anat-file to cluster (if running from local)
# [2] qsub s03_make_confounds.py
#     - If local: submitted via ssh to REMOTE_HOST
#     - If HPC:   submitted directly via qsub
# --- --- --- --- ---

# --- Parse Arguments ---
SKIP_SYNC=false
ANAT_FILE="fmriprep"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2";     shift 2 ;;
        --coreg-file)       COREG_FILE="$2";   shift 2 ;;
        --anat-file)        ANAT_FILE="$2";    shift 2 ;;
        --output-file)      OUTPUT_FILE="$2";  shift 2 ;;
        --sub)              SUBJECT="$2";      shift 2 ;;
        --ses)              SESSION="$2";      shift 2 ;;
        --skip-sync)        SKIP_SYNC=true;    shift ;;
        --help)             usage ;;
        --)  shift; EXTRA_ARGS+=("$@"); break ;;
        *)   EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# -> make subject & session labels robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]     && echo "Error: --bids-dir required"     && usage
[[ -z "$COREG_FILE" ]]   && echo "Error: --coreg-file required"   && usage
[[ -z "$OUTPUT_FILE" ]]  && echo "Error: --output-file required"  && usage
[[ -z "$SUBJECT" ]]      && echo "Error: --sub required"          && usage
[[ -z "$SESSION" ]]      && echo "Error: --ses required"          && usage

# --- Resolve paths depending on where we're running ---
if [[ "${PC_LOCATION}" == "local" ]]; then
    [[ -z "$REMOTE_BIDS_DIR" ]]   && echo "Error: \$REMOTE_BIDS_DIR not set in environment"   && exit 1
    REMOTE_HOST="${REMOTE_BIDS_DIR%%:*}"
    SUBMIT_BIDS_DIR="${REMOTE_BIDS_DIR#*:}"
else
    REMOTE_HOST=""
    SUBMIT_BIDS_DIR="$BIDS_DIR"
fi

# --- Rsync coreg + anat-only fmriprep derivatives to cluster (local only) ---
if [[ "${PC_LOCATION}" == "local" ]] && [[ "$SKIP_SYNC" != true ]]; then
    echo "Rsyncing ${COREG_FILE}"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --deriv    "$COREG_FILE"
    echo "Rsyncing ${ANAT_FILE}"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_to_hpc.sh" \
        --bids-dir "$BIDS_DIR" \
        --sub      "$SUBJECT" \
        --ses      "$SESSION" \
        --deriv    "$ANAT_FILE"
    bash "${PIPELINE_DIR}/config/hpc_helpers/rsync_code.sh"
    echo "Done copying."
else
    echo "On HPC - assuming data is already present."
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running MAKE_CONFOUNDS (aCompCor / tCompCor)"
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
echo "  Coreg file:            $COREG_FILE"
echo "  Anat file:             $ANAT_FILE"
echo "  Forwarded args:       ${EXTRA_ARGS[*]:-<none>}"
echo "-------------------------------------------------------"

# [2] Submit or run job
REMOTE_LOG_DIR="${SUBMIT_BIDS_DIR}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="make_confounds_${SUBJECT}_${SESSION}_${TIMESTAMP}"
LOG_OUT="${REMOTE_LOG_DIR}/${JOB_NAME}.o"
LOG_ERR="${REMOTE_LOG_DIR}/${JOB_NAME}.e"

RUNNER_SCRIPT="~/pipeline/functional/s03_make_confounds.py \
    --bids-dir    '${SUBMIT_BIDS_DIR}' \
    --coreg-file  '${COREG_FILE}' \
    --anat-file   '${ANAT_FILE}' \
    --output-file '${OUTPUT_FILE}' \
    --sub         '${SUBJECT}' \
    --ses         '${SESSION}' \
    --docker      '${FPREP_SIF}' \
    ${EXTRA_ARGS[*]}"

echo "-------------------------------------------------------"
echo "Submitting MAKE_CONFOUNDS job"
echo "  Subject:  $SUBJECT"
echo "  Session:  $SESSION"
echo "  Logs:     ${REMOTE_HOST:+${REMOTE_HOST}:}${LOG_OUT}"
echo "-------------------------------------------------------"
QSUB_CMD="source ~/.bash_profile; \
    source set_project.sh ${PROJ_NAME}; \
    conda activate preproc; \
    mkdir -p '${REMOTE_LOG_DIR}'; \
    qsub -V \
        -N  '${JOB_NAME}' \
        -o  '${LOG_OUT}' \
        -e  '${LOG_ERR}' \
        -l  h_rt=2:00:00 \
        -l  mem=8G \
        -pe smp 4 \
        -j  n \
        ${RUNNER_SCRIPT}"

if [[ "${PC_LOCATION}" == "local" ]]; then
    JOB_ID=$(ssh "$REMOTE_HOST" "$QSUB_CMD" | awk '{print $3}')
else
    JOB_ID=$(eval "$QSUB_CMD" | awk '{print $3}')
fi
echo "Submitted job ID: ${JOB_ID}"
