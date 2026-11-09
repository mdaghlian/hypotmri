#!/bin/bash
set -e
# --- Usage ---
usage() {
    echo "Usage: $0 --sub <ID> [--ses <ID>] [--raw] [--raw-anat] [--deriv <name>] [--bids-dir <path>] [--remote <path>]"
    echo ""
    echo "Required Arguments:"
    echo "  --sub              Subject label (e.g., sub-01)"
    echo ""
    echo "Optional:"
    echo "  --ses              Session label (e.g., ses-01). If omitted, all sessions are synced"
    echo ""
    echo "One of:"
    echo "  --raw              Sync rawdata for subject/session"
    echo "  --raw-anat         Sync only the anat/ folder for subject/session"
    echo "  --deriv            Derivative name to sync (e.g., fmriprep, freesurfer)"
    echo ""
    echo "Optional Arguments (fall back to environment variables):"
    echo "  --bids-dir         Local BIDS dir  (default: \$BIDS_DIR)"
    echo "  --remote           Remote BIDS dir (default: \$REMOTE_BIDS_DIR)"
    echo "  --dry-run          Show what would be transferred without doing it"
    echo "  --include-pattern  Only sync files matching pattern (e.g., '*_bold.nii.gz')"
    echo "  --help             Display this help message"
    exit 1
}

# --- Parse Arguments ---
DRY_RUN=""
RAW=0
RAW_ANAT=0
DERIV=""
SESSION=""
SUBJECT=""
ARG_BIDS_DIR=""
ARG_REMOTE=""
INCLUDE_PATTERN=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)        ARG_BIDS_DIR="$2";    shift 2 ;;
        --remote)          ARG_REMOTE="$2";       shift 2 ;;
        --sub)             SUBJECT="$2";          shift 2 ;;
        --ses)             SESSION="$2";          shift 2 ;;
        --raw)             RAW=1;                 shift   ;;
        --raw-anat)        RAW_ANAT=1;            shift   ;;
        --deriv)           DERIV="$2";            shift 2 ;;
        --dry-run)         DRY_RUN="--dry-run";   shift   ;;
        --include-pattern) INCLUDE_PATTERN="$2";  shift 2 ;;
        --help)            usage ;;
        *)                 echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$SUBJECT" ]]   && echo "Error: --sub required"                                                && usage

# -> make subject & session robust (empty session = all sessions)
SUBJECT="sub-${SUBJECT#sub-}"
[[ -n "$SESSION" ]] && SESSION="ses-${SESSION#ses-}"

# --- Resolve BIDS dirs (arg > env) ---
BIDS_DIR="${ARG_BIDS_DIR:-$BIDS_DIR}"
REMOTE="${ARG_REMOTE:-$REMOTE_BIDS_DIR}"

# --- Validate ---
[[ -z "$BIDS_DIR" ]]  && echo "Error: --bids_dir not set and \$BIDS_DIR not in environment"          && usage
[[ -z "$REMOTE" ]]    && echo "Error: --remote not set and \$REMOTE_BIDS_DIR not in environment"     && usage
[[ $RAW -eq 0 && $RAW_ANAT -eq 0 && -z "$DERIV" ]] \
    && echo "Error: at least one of --raw, --raw-anat, or --deriv must be specified" && usage

# --- Locate exclude file ---
EXCLUDE_FILE="${RSYNC_IGNORE:-$(dirname "$0")/.rsyncignore}"
if [[ ! -f "$EXCLUDE_FILE" ]]; then
    echo "Warning: no .rsyncignore found at $EXCLUDE_FILE, proceeding without excludes"
    EXCLUDE_OPT=""
else
    echo "Using exclude file: $EXCLUDE_FILE"
    EXCLUDE_OPT="--exclude-from=${EXCLUDE_FILE}"
fi

# --- Build include option ---
INCLUDE_OPT=()
if [[ -n "$INCLUDE_PATTERN" ]]; then
    INCLUDE_OPT=(
        --include='*/'             # descend into all subdirectories
        --include="$INCLUDE_PATTERN"  # include matching files
        --exclude='*'             # exclude everything else
    )
fi

RSYNC_OPTS="-avz --progress --ignore-existing $DRY_RUN $EXCLUDE_OPT"

# --- Helper ---
do_rsync() {
    local src="$1"
    local dst="$2"
    echo ""
    echo "  src: $src"
    echo "  dst: $dst"

    # Create destination directory - handle both local and remote paths
    if [[ "$dst" == *:* ]]; then
        local remote_host="${dst%%:*}"
        local remote_path="${dst#*:}"
        ssh "$remote_host" "mkdir -p '${remote_path}'"
    else
        mkdir -p "$dst"
    fi

    # Files that would transfer without --ignore-existing
    local all_files
    all_files=$(rsync -az --dry-run --out-format="%f" $INCLUDE_OPT "${src}/" "${dst}/" 2>/dev/null || true)

    # Files that would transfer with --ignore-existing (i.e. genuinely new)
    local new_files
    new_files=$(rsync -az --dry-run --ignore-existing --out-format="%f" $INCLUDE_OPT "${src}/" "${dst}/" 2>/dev/null || true)

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

    rsync $RSYNC_OPTS "${INCLUDE_OPT[@]}" "${src}/" "${dst}/"
}

# --- Status ---
echo "-------------------------------------------------------"
echo "Rsyncing BIDS data to cluster"
echo "-------------------------------------------------------"
echo "  Local:           $BIDS_DIR"
echo "  Remote:          $REMOTE"
echo "  Subject:         $SUBJECT"
echo "  Session:         ${SESSION:-all}"
echo "  Raw:             $([ $RAW -eq 1 ] && echo yes || echo no)"
echo "  Raw anat:        $([ $RAW_ANAT -eq 1 ] && echo yes || echo no)"
echo "  Deriv:           ${DERIV:-none}"
echo "  Dry run:         ${DRY_RUN:+yes}"
echo "  Include pattern: ${INCLUDE_PATTERN:-none}"
echo "-------------------------------------------------------"

# [1] Top-level BIDS metadata
echo "Syncing top-level BIDS metadata..."
if [[ "$REMOTE" == *:* ]]; then
    ssh "${REMOTE%%:*}" "mkdir -p '${REMOTE#*:}'"
else
    mkdir -p "$REMOTE"
fi
rsync -avz $DRY_RUN $EXCLUDE_OPT \
    --include="dataset_description.json" \
    --include=".bidsignore" \
    --include="participants.tsv" \
    --include="participants.json" \
    --exclude="*/" \
    --exclude="*" \
    "${BIDS_DIR}/" \
    "${REMOTE}/"

# Relative path below the subject dir ("" = whole subject, i.e. all sessions)
SES_REL="${SESSION:+/${SESSION}}"

# [2] Raw data (full session, or all sessions)
if [[ $RAW -eq 1 ]]; then
    echo "Syncing raw data (${SESSION:-all sessions})..."
    do_rsync \
        "${BIDS_DIR}/${SUBJECT}${SES_REL}" \
        "${REMOTE}/${SUBJECT}${SES_REL}"
fi

# [3] Raw anat only
if [[ $RAW_ANAT -eq 1 ]]; then
    echo "Syncing raw anat only..."
    if [[ -n "$SESSION" ]]; then
        ANAT_RELS=("${SUBJECT}/${SESSION}/anat")
    else
        # every ses-*/anat, or sub-XX/anat if the dataset has no session level
        ANAT_RELS=()
        for d in "${BIDS_DIR}/${SUBJECT}"/ses-*/anat "${BIDS_DIR}/${SUBJECT}/anat"; do
            [[ -d "$d" ]] && ANAT_RELS+=("${d#${BIDS_DIR}/}")
        done
        [[ ${#ANAT_RELS[@]} -eq 0 ]] && echo "  ⚠️  No anat folders found for ${SUBJECT}"
    fi
    for rel in "${ANAT_RELS[@]}"; do
        do_rsync \
            "${BIDS_DIR}/${rel}" \
            "${REMOTE}/${rel}"
    done
fi

# [4] Derivatives
if [[ -n "$DERIV" ]]; then
    if [[ "$DERIV" == "freesurfer" ]]; then
        echo "Syncing FreeSurfer derivative..."
        # FreeSurfer uses sub-##_ses-## instead of nested folders
        FS_DIR="${BIDS_DIR}/derivatives/freesurfer"
        if [[ -n "$SESSION" ]]; then
            FS_LABELS=("${SUBJECT}_${SESSION}")
        else
            FS_LABELS=()
            for d in "${FS_DIR}/${SUBJECT}"_ses-*; do
                [[ -d "$d" && ! -L "$d" ]] && FS_LABELS+=("$(basename "$d")")
            done
            [[ ${#FS_LABELS[@]} -eq 0 ]] && echo "Error: no ${SUBJECT}_ses-* dirs in ${FS_DIR}" && exit 1
        fi

        for FS_LABEL in "${FS_LABELS[@]}"; do
            do_rsync \
                "${FS_DIR}/${FS_LABEL}" \
                "${REMOTE}/derivatives/freesurfer/${FS_LABEL}"
        done

        # --- Create remote symlink sub-XX → sub-XX_ses-YY ---
        # Target: --ses if given; else mirror the local sub-XX symlink; else the only session
        REMOTE_HOST="${REMOTE%%:*}"
        REMOTE_PATH="${REMOTE#*:}"
        FS_LABEL=""
        if [[ -n "$SESSION" ]]; then
            FS_LABEL="${SUBJECT}_${SESSION}"
        elif [[ -L "${FS_DIR}/${SUBJECT}" ]]; then
            FS_LABEL="$(basename "$(readlink "${FS_DIR}/${SUBJECT}")")"
        elif [[ ${#FS_LABELS[@]} -eq 1 ]]; then
            FS_LABEL="${FS_LABELS[0]}"
        fi

        if [[ -z "$FS_LABEL" ]]; then
            echo "  ⚠️  Multiple FreeSurfer sessions and no local ${SUBJECT} symlink — skipping remote symlink"
        elif [[ -z "$DRY_RUN" ]]; then
            echo "Ensuring remote freesurfer symlink exists..."
            ssh "$REMOTE_HOST" "cd '${REMOTE_PATH}/derivatives/freesurfer' && \
                if [[ -L '${SUBJECT}' ]]; then
                    echo '  Symlink already exists: ${SUBJECT} -> $(readlink ${SUBJECT})';
                elif [[ -e '${SUBJECT}' ]]; then
                    echo '  ⚠️  ${SUBJECT} exists and is not a symlink — skipping symlink creation';
                else
                    ln -s '${FS_LABEL}' '${SUBJECT}';
                    echo '  Symlink created: ${SUBJECT} -> ${FS_LABEL}';
                fi"
        else
            echo "  [dry-run] Would ensure remote symlink: ${SUBJECT} → ${FS_LABEL}"
        fi
    else
        # Standard derivative
        echo "Syncing derivative: ${DERIV} (${SESSION:-all sessions})..."
        do_rsync \
            "${BIDS_DIR}/derivatives/${DERIV}/${SUBJECT}${SES_REL}" \
            "${REMOTE}/derivatives/${DERIV}/${SUBJECT}${SES_REL}"
    fi
fi

echo ""
echo "Done."