#!/bin/bash
set -e

# --- Usage ---
usage() {
    echo "Usage: $0 --sub <ID> [--ses <ID>] [--raw] [--deriv <name>] [--bids_dir <path>] [--remote <path>]"
    echo ""
    echo "Required Arguments:"
    echo "  --sub         Subject label (e.g., sub-01)"
    echo "  --ses         Session label (e.g., ses-01)"
    echo ""
    echo "One of:"
    echo "  --raw         Sync rawdata for subject/session"
    echo "  --deriv       Derivative name to sync (e.g., fmriprep, freesurfer)"
    echo ""
    echo "Optional Arguments (fall back to environment variables):"
    echo "  --bids_dir    Local BIDS dir  (default: \$BIDS_DIR)"
    echo "  --remote      Remote BIDS dir (default: \$REMOTE_BIDS_DIR)"
    echo "  --dry_run     Show what would be transferred without doing it"
    echo "  --help        Display this help message"
    exit 1
}

# --- Parse Arguments ---
DRY_RUN=""
RAW=0
DERIV=""
SESSION=""
SUBJECT=""
ARG_BIDS_DIR=""
ARG_REMOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids_dir)   ARG_BIDS_DIR="$2"; shift 2 ;;
        --remote)     ARG_REMOTE="$2";   shift 2 ;;
        --sub)        SUBJECT="$2";      shift 2 ;;
        --ses)        SESSION="$2";      shift 2 ;;
        --raw)        RAW=1;             shift   ;;
        --deriv)      DERIV="$2";        shift 2 ;;
        --dry_run)    DRY_RUN="--dry-run"; shift ;;
        --help)       usage ;;
        *)            echo "Unknown argument: $1"; usage ;;
    esac
done


# -> make subject & session robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# --- Resolve BIDS dirs (arg > env) ---
BIDS_DIR="${ARG_BIDS_DIR:-$BIDS_DIR}"
REMOTE="${ARG_REMOTE:-$REMOTE_BIDS_DIR}"
# --- Validate ---
[[ -z "$BIDS_DIR" ]]  && echo "Error: --bids_dir not set and \$BIDS_DIR not in environment"          && usage
[[ -z "$REMOTE" ]]    && echo "Error: --remote not set and \$REMOTE_BIDS_DIR not in environment"     && usage
[[ -z "$SUBJECT" ]]   && echo "Error: --sub required"                                                && usage
[[ -z "$SESSION" ]]   && echo "Error: --ses required"                                                && usage
[[ $RAW -eq 0 && -z "$DERIV" ]] && echo "Error: at least one of --raw or --deriv must be specified" && usage

# --- Locate exclude file ---
EXCLUDE_FILE="${RSYNC_IGNORE:-$(dirname "$0")/.rsyncignore}"
if [[ ! -f "$EXCLUDE_FILE" ]]; then
    echo "Warning: no .rsyncignore found at $EXCLUDE_FILE, proceeding without excludes"
    EXCLUDE_OPT=""
else
    echo "Using exclude file: $EXCLUDE_FILE"
    EXCLUDE_OPT="--exclude-from=${EXCLUDE_FILE}"
fi


RSYNC_OPTS="-avz --progress --ignore-existing $DRY_RUN $EXCLUDE_OPT"

# --- Helper ---
do_rsync() {
    local src="$1"
    local dst="$2"
    echo ""
    echo "  src: $src"
    echo "  dst: $dst"
    mkdir -p "$dst"

    # Files that would transfer without --ignore-existing
    local all_files
    all_files=$(rsync -az --dry-run --out-format="%f" "${src}/" "${dst}/" 2>/dev/null || true)

    # Files that would transfer with --ignore-existing (i.e. genuinely new)
    local new_files
    new_files=$(rsync -az --dry-run --ignore-existing --out-format="%f" "${src}/" "${dst}/" 2>/dev/null || true)

    # Anything in all_files but not in new_files would have been overwritten
    local conflicts
    conflicts=$(comm -23 <(echo "$all_files" | sort) <(echo "$new_files" | sort))

    if [[ -n "$conflicts" ]]; then
        echo ""
        echo "  ⚠️  The following files already exist locally and will be skipped:"
        echo "$conflicts" | while read -r f; do
            echo "      - $f"
        done
        echo ""
    fi

    rsync $RSYNC_OPTS "${src}/" "${dst}/"
}

# --- Status ---
echo "-------------------------------------------------------"
echo "Rsyncing BIDS data to cluster"
echo "-------------------------------------------------------"
echo "  Local:    $BIDS_DIR"
echo "  Remote:   $REMOTE"
echo "  Subject:  $SUBJECT"
echo "  Session:  $SESSION"
echo "  Raw:      $([ $RAW -eq 1 ] && echo yes || echo no)"
echo "  Deriv:    ${DERIV:-none}"
echo "  Dry run:  ${DRY_RUN:+yes}"
echo "-------------------------------------------------------"

# [1] Top-level BIDS metadata
echo "Syncing top-level BIDS metadata..."
rsync -avz $DRY_RUN $EXCLUDE_OPT \
    --include="dataset_description.json" \
    --include=".bidsignore" \
    --include="participants.tsv" \
    --include="participants.json" \
    --exclude="*/" \
    --exclude="*" \
    "${BIDS_DIR}/" \
    "${REMOTE}/"

# [2] Raw data
if [[ $RAW -eq 1 ]]; then
    echo "Syncing raw data..."
    do_rsync \
        "${BIDS_DIR}/${SUBJECT}/${SESSION}" \
        "${REMOTE}/${SUBJECT}/${SESSION}"
fi

# [3] Derivatives
if [[ -n "$DERIV" ]]; then
    echo "Syncing derivative: ${DERIV}..."
    do_rsync \
        "${BIDS_DIR}/derivatives/${DERIV}/${SUBJECT}/${SESSION}" \
        "${REMOTE}/derivatives/${DERIV}/${SUBJECT}/${SESSION}"
fi

echo ""
echo "Done."