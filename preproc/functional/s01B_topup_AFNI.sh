#!/bin/bash
set -e

# --- Default Values ---
SESSION="ses-01"
TASK=""
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd)

# --- Usage Function ---
usage() {
    echo "Usage: $0 --bids_dir <path> --output_dir <path> --subject <ID> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --bids_dir      Path to the BIDS root directory"
    echo "  --output_dir    Path to the output derivatives directory"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --ses           Session label (default: $SESSION)"
    echo "  --task          Task label (default: $TASK)"
    echo "  --help          Display this help message"
    exit 1
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bids_dir)     BIDS_DIR="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --sub)          SUBJECT="$2"; shift 2 ;;
        --ses)          SESSION="$2"; shift 2 ;;
        --task)         TASK="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# --- Validation ---
if [[ -z "$BIDS_DIR" || -z "$OUTPUT_DIR" || -z "$SUBJECT" ]]; then
    echo "Error: --bids_dir, --output_dir, and --subject are required."
    echo "Run with --help for details."
    exit 1
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Processing: SDC (AFNI Method)"
echo "-------------------------------------------------------"
echo " BIDS Root: $BIDS_DIR"
echo " Output:    $OUTPUT_DIR"
echo " Subject:   $SUBJECT"
echo " Session:   $SESSION"
echo " Task:      $TASK"
echo "-------------------------------------------------------"

# Construct paths
FUNC_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/func"
FMAP_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/fmap"
SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${SESSION}"
CURRENT_DIR=$PWD
# Create output directories
mkdir -p "${SUBJECT_OUTPUT_DIR}"
# remove anything inside 
cd $SUBJECT_OUTPUT_DIR

echo "=========================================="
echo "Running AFNI distortion correction"
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "Task: ${TASK}"
echo "=========================================="

# Find all the BOLD runs
BOLD_FILES=($(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}*_bold.nii*" | sort))    
if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}_task-${TASK}"
    exit 1
else
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
fi

# Process each run
run_counter=0
for BOLD in "${BOLD_FILES[@]}"; do
    run_counter=$((run_counter + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing run ${run_counter}/${#BOLD_FILES[@]}"
    echo "=========================================="
    
    # Extract run label from filename if present
    if [[ "$BOLD" =~ run-([0-9]+) ]]; then
        RUN_LABEL="run-${BASH_REMATCH[1]}"
        echo "Run: ${RUN_LABEL}"
    else
        RUN_LABEL=""
        echo "Run: (no run label)"
    fi
    
    # Create work directory for this run
    if [ -n "$RUN_LABEL" ]; then
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}"
    else
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}"
    fi
    mkdir -p "${WORK_DIR}"
    rm -rf $WORK_DIR/*
    # Extract base filenames
    BOLD_BASE="${BOLD##*/}"       
    BOLD_BASE="${BOLD_BASE%.gz}"  # Removes .gz if present
    BOLD_BASE="${BOLD_BASE%.nii}" # Removes .nii if present
    BOLD_BASE="${BOLD_BASE%_bold}" # Removes _bold if present

    # Find corresponding reverse-PE (TOPUP) and SBREF files
    if [ -n "$RUN_LABEL" ]; then
        TOPUP=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*epi.nii*" | head -n 1)
        SBREF=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*sbref.nii*" | head -n 1)
    else
        TOPUP=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*epi.nii*" | grep -v "run-" | head -n 1)
        SBREF=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*sbref.nii*" | grep -v "run-" | head -n 1)
    fi
    
    # Get number of volumes
    nvolsTP=$(fslnvols "$TOPUP")
    nvolsB=$(fslnvols "$BOLD")
    
    echo ""
    echo "Volume information:"
    echo "  BOLD volumes: ${nvolsB}"
    echo "  Reverse-PE volumes: ${nvolsTP}"
    
    # Read phase encoding direction from jsons
    BOLD_JSON="${BOLD%.nii*}.json"
    TOPUP_JSON="${TOPUP%.nii*}.json"
    BOLD_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${BOLD_JSON}" | cut -d'"' -f4)
    TOPUP_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${TOPUP_JSON}" | cut -d'"' -f4)
    
    echo ""
    echo "Phase encoding parameters:"
    echo "  BOLD PE direction: ${BOLD_PE}"
    echo "  Reverse-PE direction: ${TOPUP_PE}"
    
    # **** AFNI CONVERT  ****

    echo ""
    echo "Converting BOLD to AFNI format..."
    if [[ "$BOLD" == *.gz ]]; then
        gunzip -c "$BOLD" > "${WORK_DIR}/bold_temp.nii"
        3dcopy "${WORK_DIR}/bold_temp.nii" "${WORK_DIR}/bold+orig"
        rm "${WORK_DIR}/bold_temp.nii"
    else
        3dcopy "$BOLD" "${WORK_DIR}/bold+orig"
    fi
    
    # Convert reverse-PE to AFNI format
    echo "Converting reverse-PE to AFNI format..."
    if [[ "$TOPUP" == *.gz ]]; then
        gunzip -c "$TOPUP" > "${WORK_DIR}/reverse_temp.nii"
        3dcopy "${WORK_DIR}/reverse_temp.nii" "${WORK_DIR}/reverse+orig"
        # rm "${WORK_DIR}/reverse_temp.nii"
    else
        3dcopy "$TOPUP" "${WORK_DIR}/reverse+orig"
    fi
    
    # Calculate volume indices for AFNI
    # Extract last N volumes from BOLD to match reverse-PE volumes
    start_idx=$(($nvolsB - $nvolsTP))
    end_idx=$(($nvolsB - 1))
    IdxEPI="[${start_idx}..${end_idx}]"
    IdxRev="[0..$((nvolsTP - 1))]"
    
    echo ""
    echo "Running AFNI unWarpEPIfloat.py..."
    echo "  Last ${nvolsTP} BOLD images: ${IdxEPI}"
    echo "  Reverse-PE images: ${IdxRev}"

    # Run AFNI's distortion correction
    # -f forward -r reverse -d  
    cd $WORK_DIR   
    # python ./unWarpEPIfloat.py \
    python "$SCRIPT_DIR/unWarpEPIfloat.py" \
        -f "bold+orig${IdxEPI}" \
        -r "reverse+orig${IdxRev}" \
        -d "bold" \
        -s "${BASE_NAME}"
    
    echo "Extracting corrected BOLD data..."
    # The unWarpEPIfloat.py output is in unWarpOutput_*/06_*_HWV.nii.gz
    UNWARP_OUTPUT=$(find "${WORK_DIR}/unWarpOutput_TS" -name "06_*_HWV.nii.gz" | head -n 1)
    if [ ! -f "$UNWARP_OUTPUT" ]; then
        echo "Error: AFNI unwarp output not found!"
        exit 1
    fi
    
    # Copy the unwarped output to final location
    cp "$UNWARP_OUTPUT" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_bold_sdc.nii.gz"
    
    echo "Applying distortion correction to SBREF..."
    # For SBREF, we need to apply the same correction
    # Convert SBREF to AFNI format
    if [[ "$SBREF" == *.gz ]]; then
        gunzip -c "$SBREF" > "${WORK_DIR}/sbref_temp.nii"
        3dcopy "${WORK_DIR}/sbref_temp.nii" "${WORK_DIR}/sbref+orig"
        rm "${WORK_DIR}/sbref_temp.nii"
    else
        3dcopy "$SBREF" "${WORK_DIR}/sbref+orig"
    fi
    
    # Apply the same warp to SBREF using the displacement field from the BOLD correction
    # The warp is stored in the unWarpOutput directory
    WARP_FILE=$(find "${WORK_DIR}/unWarpOutput_TS" -name "*_WARP.nii.gz" | head -n 1)
    
    if [ -f "$WARP_FILE" ]; then
        3dNwarpApply \
            -source "${WORK_DIR}/sbref+orig" \
            -nwarp "$WARP_FILE" \
            -prefix "${WORK_DIR}/sbref_sdc.nii.gz"
        
        cp "${WORK_DIR}/sbref_sdc.nii.gz" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_sbref_sdc.nii.gz"
    else
        echo "Warning: Warp file not found for SBREF correction"
        echo "Copying original SBREF as placeholder"
        cp "$SBREF" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_sbref_sdc.nii.gz"
    fi
    
    # Optional: Clean up working directory
    # Uncomment to remove intermediate files
    # echo "Cleaning up intermediate files..."
    # rm -f "${WORK_DIR}"/*+orig.*
    # rm -rf "${WORK_DIR}/unWarpOutput_unwarp_out"
    
    echo "Run ${run_counter} completed!"
    
done

echo ""
echo "=========================================="
echo "All Runs Completed Successfully!"
echo "=========================================="
echo "Processed ${run_counter} run(s)"
echo "Output directory: ${SUBJECT_OUTPUT_DIR}"
echo "Done!"
cd $CURRENT_DIR