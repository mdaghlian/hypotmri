#!/bin/bash
#$ -S /bin/bash
#$ -V
#$ -cwd
set -e

# --- Usage Function ---
usage() {
    echo "Usage: qc_surfaces.sh <sub> "
    echo "Opens subject surfaces in freeview for QC"
    echo "Assumes SUBJECTS_DIR is set in environment"
    exit 1
}

if [[ $# -lt 1 ]] ; then
    usage
fi
# A simple script to open a FreeSurfer subject in freeview.
# It loads T1, T2 (if available), and the pial and white surfaces.
SUBJECT=$1
SUBJECT="sub-${SUBJECT#sub-}"

# Define paths to key files
DIR_MRI="$SUBJECTS_DIR/$SUBJECT/mri"
DIR_SURF="$SUBJECTS_DIR/$SUBJECT/surf"
T1_FILE="$DIR_MRI/orig.mgz"
OTHER_VOLS=("brainmask" "rawavg" "T2" )

# Check if the subject directory exists
if [ ! -d "$SUBJECTS_DIR/$SUBJECT" ]; then
    echo "Error: Subject directory not found for $SUBJECT in $SUBJECTS_DIR"
    exit 1
fi

# Build the base freeview command
freeview_cmd="freeview -v $T1_FILE"

for file in "${OTHER_VOLS[@]}"; do
    FILE_PATH="$DIR_MRI/${file}.mgz"
    if [ -f "$FILE_PATH" ]; then
        freeview_cmd+=" $FILE_PATH"
    fi
done

# Add surfaces with specified colors
freeview_cmd+=" -f $DIR_SURF/lh.pial:edgecolor=blue \
$DIR_SURF/rh.pial:edgecolor=blue \
$DIR_SURF/lh.white:edgecolor=yellow \
$DIR_SURF/rh.white:edgecolor=yellow"

# Execute the command
echo "Opening subject $SUBJECT in freeview..."
eval $freeview_cmd &