#!/bin/bash
set -e

# --- Usage ---
usage() {
    echo "Usage: $0 --sub <ID> [--ses <ID>] [--deriv <name>] [--bids-dir <path>] [--remote <path>]"
    echo ""
    echo "Required Arguments:"
    echo "  --sub         Subject label (e.g., sub-01)"
    echo "  --ses         Session label (e.g., ses-01)"
    echo ""
    echo "One of:"
    echo "  --deriv       Derivative name to sync (e.g., fmriprep, freesurfer)"
    echo ""
    echo "Optional Arguments (fall back to environment variables):"
    echo "  --bids-dir    Local BIDS dir  (default: \$BIDS_DIR)"
    echo "  --remote      Remote BIDS dir (default: \$REMOTE_BIDS_DIR)"
    echo "  --dry-run     Show what would be transferred without doing it"
    echo "  --help        Display this help message"
    exit 1
}

# --- Parse Arguments ---
DRY_RUN=""
DERIV=""
SESSION=""
SUBJECT=""
ARG_BIDS_DIR=""
ARG_REMOTE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)   ARG_BIDS_DIR="$2"; shift 2 ;;
        --remote)     ARG_REMOTE="$2";   shift 2 ;;
        --sub)        SUBJECT="$2";      shift 2 ;;
        --ses)        SESSION="$2";      shift 2 ;;
        --deriv)      DERIV="$2";        shift 2 ;;
        --dry-run)    DRY_RUN="--dry-run"; shift ;;
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
[[ -z "$BIDS_DIR" ]]  && echo "Error: --bids-dir not set and \$BIDS_DIR not in environment" && usage
[[ -z "$REMOTE" ]]    && echo "Error: --remote not set and \$REMOTE_BIDS_DIR not in environment" && usage
[[ -z "$SUBJECT" ]]   && echo "Error: --sub required" && usage
[[ -z "$SESSION" ]]   && echo "Error: --ses required" && usage
[[ -z "$DERIV" ]]     && echo "Error: --deriv must be specified" && usage

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
echo "Rsyncing BIDS data from cluster"
echo "-------------------------------------------------------"
echo "  Remote:   $REMOTE"
echo "  Local:    $BIDS_DIR"
echo "  Subject:  $SUBJECT"
echo "  Session:  $SESSION"
echo "  Deriv:    ${DERIV}"
echo "  Dry run:  ${DRY_RUN:+yes}"
echo "-------------------------------------------------------"

# [1] Derivatives
if [[ "$DERIV" == "freesurfer" ]]; then
    # FreeSurfer stores output as sub-##_ses-## rather than sub-##/ses-##
    FS_LABEL="${SUBJECT}_${SESSION}"
    FS_LOCAL="${BIDS_DIR}/derivatives/freesurfer"

    echo "Syncing FreeSurfer derivative: ${FS_LABEL}..."
    do_rsync \
        "${REMOTE}/derivatives/freesurfer/${FS_LABEL}" \
        "${FS_LOCAL}/${FS_LABEL}"

    # Create sub-## symlink -> sub-##_ses-## if it doesn't already exist
    SYMLINK="${FS_LOCAL}/${SUBJECT}"
    if [[ -z "$DRY_RUN" ]]; then
        if [[ -L "$SYMLINK" ]]; then
            echo "  Symlink already exists: $SYMLINK -> $(readlink "$SYMLINK")"
        elif [[ -e "$SYMLINK" ]]; then
            echo "  ⚠️  $SYMLINK exists and is not a symlink — skipping symlink creation"
        else
            ln -s "${FS_LABEL}" "$SYMLINK"
            echo "  Symlink created: $SYMLINK -> ${FS_LABEL}"
        fi
    else
        echo "  [dry-run] Would create symlink: $SYMLINK -> ${FS_LABEL}"
    fi
else
    echo "Syncing derivative: ${DERIV}..."
    do_rsync \
        "${REMOTE}/derivatives/${DERIV}/${SUBJECT}/${SESSION}" \
        "${BIDS_DIR}/derivatives/${DERIV}/${SUBJECT}/${SESSION}"
fi

echo ""
echo "Done."