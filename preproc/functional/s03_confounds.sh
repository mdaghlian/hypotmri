#!/bin/bash
set -e

# --- Default Values ---
SESSION="ses-01"
TASK_LIST=("pRFLE" "pRFRE")
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd)

# --- Usage Function ---
usage() {
    echo "Usage: $0 --input_dir <path> --output_dir <path> --subject <ID> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --input_dir     Path to the bold + sbref files"
    echo "  --bids_dir    Path to the output derivatives directory"
    echo "  --bids_filter_file  Path to bids_filter_file " 
    echo "  --sub           Subject label (e.g., sub-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)        INPUT_DIR="$2"; shift 2 ;;
        --bids_dir)         BIDS_DIR="$2"; shift 2 ;;
        --bids_filter_file)  BIDS_FILTER_FILE="$2"; shift 2;;
        --sub)              SUBJECT="$2"; shift 2 ;;
        --help)             usage ;;
        *)                  echo "Unknown argument: $1"; usage ;;
    esac
done


# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Running fmriprep - to get the confounds "
echo "-------------------------------------------------------"
echo " Input:     $INPUT_DIR"
echo " Output:    $BIDS_DIR"
echo " Subject:   $SUBJECT"
echo "-------------------------------------------------------"

# Construct paths
FPREP_SES=$BIDS_DIR/$SUBJECT/ses-fprep/func
mkdir -p "${FPREP_SES}"

# Find all the BOLD runs
# sub-hp01_ses-01_task-pRFRE_run-03_bold_sdc_space-fsT1_desc-moco_bbreg_bold
BOLD_FILES=($(find "${INPUT_DIR}" -name "${SUBJECT}_*space-fsT1_*.nii*" | sort))    
if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}_task-${TASK}"
    exit 1
else
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
fi

# Process each run
run_counter=0
for BOLD_FILE in "${BOLD_FILES[@]}"; do
    run_counter=$((run_counter + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing run ${run_counter}/${#BOLD_FILES[@]}"
    echo "=========================================="
    
    # Extract run label from filename if present
    if [[ "$BOLD_FILE" =~ run-([0-9]+) ]]; then
        RUN_LABEL="run-${BASH_REMATCH[1]}"
        echo "Run: ${RUN_LABEL}"
    else
        RUN_LABEL=""
        echo "Run: (no run label)"
    fi

    # Extract task label 
    if [[ "$BOLD_FILE" =~ task-([a-zA-Z0-9]+) ]]; then
        TASK_LABEL="task-${BASH_REMATCH[1]}"
        echo "Task: ${TASK_LABEL}"
    else
        TASK_LABEL="task-unknown"
        echo "Task: (no task label found)"
    fi
    FPREP_BOLD="${FPREP_SES}/${SUBJECT}_ses-fprep_${TASK_LABEL}_${RUN_LABEL}_bold.nii.gz"
    FPREP_JSON="${FPREP_SES}/${SUBJECT}_ses-fprep_${TASK_LABEL}_${RUN_LABEL}_bold.json"
    BOLD_TR=$(fslval $BOLD_FILE pixdim4)
cat <<EOF > "$FPREP_JSON"
{
  "RepetitionTime": $BOLD_TR,
  "TaskName": "$TASK_LABEL"
}
EOF
    cp $BOLD_FILE $FPREP_BOLD
    
done

echo ""
echo "=========================================="
echo "Copied it all over - now for fmriprep"
echo "=========================================="

fmriprep-docker \
  $BIDS_DIR \
  $BIDS_DIR/derivatives/fmriprep \
  participant \
  --participant-label $SUBJECT \
  --fs-subjects-dir  $SUBJECTS_DIR \
  --fs-license-file /Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/code/license.txt \
  --skip-bids-validation \
  --omp-nthreads 8 \
  --output-spaces func T1w fsnative \
  -w $BIDS_DIR/../BIDSWF \
  --bids-filter-file $BIDS_FILTER_FILE --session-label fprep