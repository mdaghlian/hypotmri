#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
set -e

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids-dir <path> --sub <sub> --ses <ses>"
    echo ""
    echo "Required Arguments:"
    echo "  --bids-dir      Path BIDS directory "
    echo "  --sub           Subject label (e.g., sub-01)"
    echo "  --ses           Session label (e.g., ses-01)"
    echo "  --suffix        Suffix for folders, if want to test without overwrite"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}
# --- SCRTIPT OVERVIEW ---
# [1] Create FPREPBIDS -> inside BIDS_DIR/derivatives
# -- Why? We don't want to run fmriprep on the actual "raw" data
# -- but rather on a subset of the data we have preprocessed. To
# -- do this we put only what we want inside "FPREPBIDS"
# -- The fMRIPREP+Freesurfer outputs are placed in the usual place
# -- (BIDS_DIR/derivatives)
# [2] Symlink the anatomical (T1w) from BIDS_DIR/SUBJECT/SESSION/anat -
# -- rather than copying it, so it doesn't exist twice on disk. The
# -- container is bind-mounted with access to BIDS_DIR itself (at the
# -- same absolute path) so these symlinks resolve inside it too.
# [3] Run fmriprep
# --- --- --- ---
SUFFIX=""
# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids-dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --ses)              SESSION="$2"; shift 2 ;;
        --suffix)           SUFFIX="$2"; shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# -> make subject & session robust
SUBJECT="sub-${SUBJECT#sub-}"
SESSION="ses-${SESSION#ses-}"

# -> resolve BIDS_DIR to an absolute path. This matters for the symlink
# strategy below: symlink targets are written as absolute paths, and the
# container is later bind-mounted with BIDS_DIR at that *same* absolute
# path, so the two have to agree on what "absolute" means here.
BIDS_DIR="$(cd "$BIDS_DIR" && pwd)"

# Symlink every file under $1 into $2, mirroring subdirectories as real
# directories rather than symlinking the directory itself. This matters
# because pybids-based tools (fMRIPrep uses pybids internally) don't
# reliably descend into symlinked *directories* during BIDS discovery -
# symlinking only the files, inside real directories, avoids anat scans
# silently going undiscovered.
symlink_tree() {
    local src_dir="$1"
    local dst_dir="$2"
    local file rel dst_file
    while IFS= read -r -d '' file; do
        rel="${file#"$src_dir"/}"
        dst_file="${dst_dir}/${rel}"
        mkdir -p "$(dirname "$dst_file")"
        ln -s "$(cd "$(dirname "$file")" && pwd)/$(basename "$file")" "$dst_file"
    done < <(find "$src_dir" -type f -print0)
}

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - for anatomy "
echo "-------------------------------------------------------"
echo " BIDS DIR:    $BIDS_DIR"
echo " Subject:   $SUBJECT"
echo " Session:   $SESSION"
echo "-------------------------------------------------------"

# [1] Create FPREP BIDS
FPREP_BIDS_DIR="${BIDS_DIR}/derivatives/FPREP_BIDS${SUFFIX}"
if [[ ! -d "${FPREP_BIDS_DIR}" ]]; then
    mkdir -p ${FPREP_BIDS_DIR}
fi 
FPREP_BIDS_DIR_WF="${BIDS_DIR}/derivatives/FPREP_BIDS_WF${SUFFIX}"
if [[ ! -d "${FPREP_BIDS_DIR_WF}" ]]; then
    mkdir -p $FPREP_BIDS_DIR_WF
fi 
# -> Create bids json if it doesn't exist
BIDS_JSON="${FPREP_BIDS_DIR}/dataset_description.json"
if [[ ! -f "${BIDS_JSON}" ]]; then
    printf "{\"Name\": \"Example dataset\", \"BIDSVersion\": \"1.0.2\"}" >> "$BIDS_JSON"
fi

# -> Create freesurfer output, if it doesn't exist
# Note this is inside the "true" BIDS_DIR 
SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer${SUFFIX}"
if [[ ! -d "${SUBJECTS_DIR}" ]]; then
    mkdir -p "${SUBJECTS_DIR}"
fi

FPREP_SES="${FPREP_BIDS_DIR}/${SUBJECT}/${SESSION}"
if [[ -e "${FPREP_SES}" ]]; then
    rm -rf ${FPREP_SES}
fi
mkdir -p "${FPREP_SES}"

echo "Symlinking anatomy (avoids duplicating the T1w on disk)"
ANAT_SRC="${BIDS_DIR}/${SUBJECT}/${SESSION}/anat"
symlink_tree "${ANAT_SRC}" "${FPREP_SES}/anat"
echo "running fprep in ${FPREP_BIDS_DIR} and ${FPREP_SIF}"
FPREP_OUT="${BIDS_DIR}/derivatives/fmriprep${SUFFIX}"
[[ ! -d "${FPREP_OUT}" ]] && mkdir -p "${FPREP_OUT}"

if [[ "$CONTAINER_TYPE" == "docker" ]]; then
    docker run --rm \
      -v $FPREP_BIDS_DIR:/data:ro \
      -v $BIDS_DIR:$BIDS_DIR:ro \
      -v $FPREP_OUT:/out \
      -v $FPREP_BIDS_DIR_WF:/work \
      -v $SUBJECTS_DIR:/fsdir \
      -v $PIPELINE_DIR/config/license.txt:/license.txt \
      $FPREP_IMAGE \
        /data /out participant \
        --participant-label $SUBJECT \
        --skip_bids_validation \
        --fs-subjects-dir /fsdir \
        --fs-license-file /license.txt \
        --work-dir /work \
        --anat-only \
        --omp-nthreads 8 --nprocs 8

elif [[ "$CONTAINER_TYPE" == "apptainer" || "$CONTAINER_TYPE" == "singularity" ]]; then
    ${CONTAINER_TYPE} run \
      --cleanenv \
      -B $FPREP_BIDS_DIR:/data \
      -B $BIDS_DIR:$BIDS_DIR:ro \
      -B $FPREP_OUT:/out \
      -B $FPREP_BIDS_DIR_WF:/work \
      -B $SUBJECTS_DIR:/fsdir \
      -B $PIPELINE_DIR/config/license.txt:/license.txt \
      $SIF_DIR/$FPREP_SIF \
        /data /out participant \
        --participant-label $SUBJECT \
        --skip_bids_validation \
        --fs-subjects-dir /fsdir \
        --fs-license-file /license.txt \
        --work-dir /work \
        --anat-only \
        --omp-nthreads 8 --nprocs 8
else
    echo "Invalid CONTAINER_TYPE: $CONTAINER_TYPE"
    exit 1
fi
# Create a symlink between subject dir and the annoying way that fmriprep does 
# freesurfer naming. See:
# https://neurostars.org/t/automatic-freesurfer-subject-name-includes-session-tag/35135/2
# https://github.com/nipreps/fmriprep/pull/3588
SUB_FS=$SUBJECTS_DIR/$SUBJECT
SUB_FS_FPREP=$SUBJECTS_DIR/${SUBJECT}_${SESSION}
if [[ ! -d "${SUB_FS}" ]]; then
    ln -s $SUB_FS_FPREP $SUB_FS
fi