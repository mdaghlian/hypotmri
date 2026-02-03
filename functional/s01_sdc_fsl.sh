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
echo "Processing: SDC"
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

# Create output directories
mkdir -p "${SUBJECT_OUTPUT_DIR}"

echo "=========================================="
echo "Running topup correction"
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

# Convert PE direction to FSL format
convert_pe_to_vector() {
    case $1 in
        "j-") echo "0 -1 0" ;;
        "j") echo "0 1 0" ;;
        "i-") echo "-1 0 0" ;;
        "i") echo "1 0 0" ;;
        "k-") echo "0 0 -1" ;;
        "k") echo "0 0 1" ;;
        *) echo "Error: Unknown PE direction: $1"; exit 1 ;;
    esac
}

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
    
    # Extract base filenames
    BOLD_BASE="${BOLD##*/}"       
    BOLD_BASE="${BOLD_BASE%.gz}"  # Removes .gz if present
    BOLD_BASE="${BOLD_BASE%.nii}" # Removes .nii if present
    BOLD_BASE="${BOLD_BASE%_bold}" # Removes _bold if present

    # Find corresponding TOPUP and SBREF files
    if [ -n "$RUN_LABEL" ]; then
        TOPUP=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*epi.nii*" | head -n 1)
        SBREF=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*sbref.nii*" | head -n 1)
    else
        TOPUP=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*epi.nii*" | grep -v "run-" | head -n 1)
        SBREF=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*sbref.nii*" | grep -v "run-" | head -n 1)
    fi
    
    # Create the TOPUP pair that fsl needs
    echo "Creating TOPUP pair (mean images)..."
    nvolsTP=$(fslnvols "$TOPUP")
    nvolsB=$(fslnvols "$BOLD")
    start=$(($nvolsB - $nvolsTP))
    # Use last N vols of bold paired with TOPUP file 
    fslroi "${BOLD}"   "${WORK_DIR}/fw.nii.gz" $start $nvolsTP
    fslroi "${TOPUP}" "${WORK_DIR}/bw.nii.gz"  0 $nvolsTP

    # Merge into 4D with 2 volumes
    fslmerge -t "${WORK_DIR}/fw_bw_pair.nii.gz" \
        "${WORK_DIR}/fw.nii.gz" \
        "${WORK_DIR}/bw.nii.gz"

    
    # Read phase encoding direction & total readout time from jsons
    BOLD_JSON="${BOLD%.nii*}.json"
    TOPUP_JSON="${TOPUP%.nii*}.json"
    BOLD_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${BOLD_JSON}" | cut -d'"' -f4)
    TOPUP_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${TOPUP_JSON}" | cut -d'"' -f4)
    BOLD_TRT=$(grep -o '"TotalReadoutTime"[[:space:]]*:[[:space:]]*[0-9.]*' "${BOLD_JSON}" | awk '{print $2}')
    TOPUP_TRT=$(grep -o '"TotalReadoutTime"[[:space:]]*:[[:space:]]*[0-9.]*' "${TOPUP_JSON}" | awk '{print $2}')
    echo ""
    echo "Phase encoding parameters:"
    echo "  BOLD PE direction: ${BOLD_PE}"
    echo "  TOPUP PE direction: ${TOPUP_PE}"
    echo "  BOLD Total Readout Time: ${BOLD_TRT}"
    echo "  TOPUP Total Readout Time: ${TOPUP_TRT}"
    
    BOLD_PE_VEC=$(convert_pe_to_vector "${BOLD_PE}")
    TOPUP_PE_VEC=$(convert_pe_to_vector "${TOPUP_PE}")
    echo "Writing TOPUP acquisition parameters..."
    # 1 row per volume, in order 
    # gives the important info for fsl topup
    {
    for i in $(seq 1 "$nvolsTP"); do
        echo "${BOLD_PE_VEC} ${BOLD_TRT}"
    done
    for i in $(seq 1 "$nvolsTP"); do
        echo "${TOPUP_PE_VEC} ${TOPUP_TRT}"
    done
    } > "${WORK_DIR}/acqparams.txt"

    echo " Running TOPUP..."
    topup \
      --imain="${WORK_DIR}/fw_bw_pair.nii.gz" \
      --datain="${WORK_DIR}/acqparams.txt" \
      --config=b02b0.cnf \
      --out="${WORK_DIR}/topup_results" 

    echo "Applying TOPUP to full BOLD time series & BREF..."
    
    applytopup \
        --imain="${SBREF}" \
        --datain="${WORK_DIR}/acqparams.txt" \
        --inindex=1 \
        --topup="${WORK_DIR}/topup_results" \
        --method=jac \
        --out="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_sdc_sbref.nii.gz"

    applytopup \
        --imain="${BOLD}" \
        --datain="${WORK_DIR}/acqparams.txt" \
        --inindex=1 \
        --topup="${WORK_DIR}/topup_results" \
        --method=jac \
        --out="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_sdc_bold.nii.gz"
    # Optional: Clean up working directory
    # Uncomment to remove intermediate files
    # rm -rf "${WORK_DIR}"
    
done

echo ""
echo "=========================================="
echo "All Runs Completed Successfully!"
echo "=========================================="
echo "Processed ${run_counter} run(s)"
echo "Output directory: ${SUBJECT_OUTPUT_DIR}"
echo "Done!"