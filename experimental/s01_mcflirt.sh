#!/bin/bash
#
# Mcflirt - run motion correction using FSL's mcflirt, save transformation matrices
#
# Usage: ./s01_mcflirt.sh <bids_dir> <output_dir> <subject> <session> <task> [run1,run2,... or 'all']
#

set -e  # Exit on error

# Check for required arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <bids_dir> <output_dir> <subject> <session> <task> [run1,run2,... or 'all']"
    echo "Example: $0 /data/bids /data/derivatives/preproc sub-01 ses-01 rest all"
    exit 1
fi

BIDS_DIR=$1
OUTPUT_DIR=$2
SUBJECT=$3
SESSION=$4
TASK=$5
RUN_SPEC=${6:-"all"}

# Construct paths
FUNC_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/func"
SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${SESSION}"

# Create output directories
mkdir -p "${SUBJECT_OUTPUT_DIR}"

echo "=========================================="
echo "Running Motion Correction with MCFLIRT"
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "Task: ${TASK}"
echo "=========================================="

# Determine which runs to process
if [ "$RUN_SPEC" = "all" ]; then
    BOLD_FILES=($(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_run-*_bold.nii*" | sort))
    
    if [ ${#BOLD_FILES[@]} -eq 0 ]; then
        BOLD_FILES=($(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*bold.nii*" | grep -v "run-" | head -n 1))
    fi
    
    if [ ${#BOLD_FILES[@]} -eq 0 ]; then
        echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}_task-${TASK}"
        exit 1
    fi
    
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
else
    IFS=',' read -ra RUNS <<< "$RUN_SPEC"
    BOLD_FILES=()
    for run in "${RUNS[@]}"; do
        run=$(echo "$run" | xargs)
        bold_file=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${run}_bold.nii*" | head -n 1)
        if [ -z "$bold_file" ]; then
            echo "Warning: BOLD file not found for ${run}, skipping..."
        else
            BOLD_FILES+=("$bold_file")
        fi
    done
    
    if [ ${#BOLD_FILES[@]} -eq 0 ]; then
        echo "Error: No valid BOLD files found for specified runs"
        exit 1
    fi
    
    echo "Processing ${#BOLD_FILES[@]} specified run(s)"
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
    
    # Create work directory for this run
    if [ -n "$RUN_LABEL" ]; then
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}"
    else
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/work"
    fi
    mkdir -p "${WORK_DIR}"
    
    # Find corresponding SBREF files
    if [ -n "$RUN_LABEL" ]; then
        SBREF_FILE=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*sbref.nii*" | head -n 1)
    else
        SBREF_FILE=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*sbref.nii*" | grep -v "run-" | head -n 1)
    fi
    
    echo "Found files:"
    echo "  BOLD: ${BOLD_FILE}"
    if [ -n "$SBREF_FILE" ]; then
        echo "  SBREF: ${SBREF_FILE}"
    fi
    
    # Extract base filenames
    BOLD_BASE=$(basename "${BOLD_FILE}" .nii.gz)
    BOLD_BASE=$(basename "${BOLD_BASE}" .nii)
    
    # Read phase encoding direction from JSON sidecar files
    BOLD_JSON="${BOLD_FILE%.nii*}.json"
    
    if [ ! -f "$BOLD_JSON" ] ; then
        echo "Error: JSON sidecar(s) not found"
        echo "Skipping this run..."
        continue
    fi
    
    # Extract phase encoding info from JSON
    BOLD_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${BOLD_JSON}" | cut -d'"' -f4)
    BOLD_TRT=$(grep -o '"TotalReadoutTime"[[:space:]]*:[[:space:]]*[0-9.]*' "${BOLD_JSON}" | awk '{print $2}')
    
    echo ""
    echo "Phase encoding parameters:"
    echo "  BOLD PE direction: ${BOLD_PE}"
    echo "  BOLD Total Readout Time: ${BOLD_TRT}"
    
    BOLD_PE_VEC=$(convert_pe_to_vector "${BOLD_PE}")
    
    # ============================================================
    # STEP 1: MOTION CORRECTION (using mcflirt)
    # ============================================================
    
    MCFLIRT_OUTPUT="${WORK_DIR}/bold_mcf"
    MCFLIRT_PARAMS="${WORK_DIR}/bold_mcf.par"
    
    # Determine reference volume for motion correction
    if [ -n "$SBREF_FILE" ]; then
        echo "Using SBREF as motion correction reference"
        mcflirt -in "${BOLD_FILE}" \
                -reffile "${SBREF_FILE}" \
                -out "${MCFLIRT_OUTPUT}" \
                -mats \
                -plots \
                -report
    else
        echo "Using median volume as motion correction reference"
        mcflirt -in "${BOLD_FILE}" \
                -out "${MCFLIRT_OUTPUT}" \
                -mats \
                -plots \
                -report
    fi
    
    echo "Motion correction completed"
    echo "  Output: ${MCFLIRT_OUTPUT}"
    echo "  Motion parameters: ${MCFLIRT_PARAMS}"
    
    # Save motion parameters to output directory
    MOTION_PARAMS_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt_motion.par"
    cp "${MCFLIRT_PARAMS}" "${MOTION_PARAMS_OUTPUT}"
    
    # Generate motion plots
    if [ -f "${WORK_DIR}/bold_mcf_abs.rms" ]; then
        MOTION_PLOTS_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt_motion_plots"
        cp "${WORK_DIR}/bold_mcf_abs.rms" "${MOTION_PLOTS_OUTPUT}_abs.rms"
        cp "${WORK_DIR}/bold_mcf_rel.rms" "${MOTION_PLOTS_OUTPUT}_rel.rms"
        cp "${WORK_DIR}/bold_mcf_abs_mean.rms" "${MOTION_PLOTS_OUTPUT}_abs_mean.rms"
        cp "${WORK_DIR}/bold_mcf_rel_mean.rms" "${MOTION_PLOTS_OUTPUT}_rel_mean.rms"
    fi
    
    cp "${MCFLIRT_OUTPUT}.nii.gz" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-boldmc.nii.gz"
    # cp "${WORK_DIR}/bold_mcf.nii.gz.mat" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt_transforms.mat"
    
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