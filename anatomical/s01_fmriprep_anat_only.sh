#!/bin/bash
set -e

# --- Default Values ---
SESSION="ses-01"
FPREP_SES
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd)

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids_dir <path> --sub <ID> "
    echo ""
    echo "Required Arguments:"
    echo "  --bids_dir    Path to the output derivatives directory"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids_dir)         BIDS_DIR="$2"; shift 2 ;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - for anatomy "
echo "-------------------------------------------------------"
echo " Output:    $BIDS_DIR"
echo " Subject:   $SUBJECT"
echo "-------------------------------------------------------"

# [1] Create .bidsignore if doesn't exist
BIDS_IGNORE="${BIDS_DIR}/.bidsignore"
if [[ ! -f "${BIDS_IGNORE}" ]]; then
    printf "**/ses-01/func/\n**/ses-01/fmap/\n" >> "$BIDS_IGNORE"
fi

# Now create the fprep session
# -dummy session used later
# -we want the anatomy to be used here too
FPREP_SES="${BIDS_DIR}/${SUBJECT}/ses-fprep"
if [[ -e "${FPREP_SES}" ]]; then
    rm -rf ${FPREP_SES}
fi
mkdir -p "${FPREP_SES}"
echo "Copying anatomy to ses-fprep, to make fmriprep happy" 
ANAT_SRC="${BIDS_DIR}/${SUBJECT}/ses-01/anat"
ANAT_DEST="${BIDS_DIR}/${SUBJECT}/ses-fprep/anat"
mkdir ${ANAT_DEST}
echo $ANAT_SRC
for file in "$ANAT_SRC"/*; do 
    if [ -f "$file" ]; then
        # Get the base filename without the path
        basename=$(basename "$file")
        # Create the new filename by replacing 'ses-01' with 'ses-fprep'
        new_name="${basename//ses-01/ses-fprep}"
        # Copy the file to the new location with the new name
        cp "$file" "$ANAT_DEST/$new_name"
        echo "Copied: $basename -> $new_name"
    fi
done

fmriprep-docker \
  $BIDS_DIR \
  $BIDS_DIR/derivatives/fmriprep \
  participant \
  --participant-label $SUBJECT \
  --fs-subjects-dir  $SUBJECTS_DIR \
  --fs-license-file $BIDS_DIR/code/license.txt \
  -w $BIDS_DIR/../BIDSWF --anat-only \
  --omp-nthreads 8 --session-label fprep

# Create a symlink between subject dir and the annoying way that fmriprep does 
# freesurfer naming
SUB_FS=$SUBJECTS_DIR/$SUBJECT
SUB_FS_FPREP=$SUBJECTS_DIR/${SUBJECT}_ses-fprep
if [[ ! -d "${SUB_FS}" ]]; then
    ln -s $SUB_FS_FPREP $SUB_FS
fi
