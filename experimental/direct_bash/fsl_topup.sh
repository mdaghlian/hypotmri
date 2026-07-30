#!/bin/bash
#
# FSL Topup Correction Script for BIDS Data (Multi-Run Support)
# This script applies topup correction to fMRI data using opposite phase-encoded EPI scans
# Supports processing multiple runs in a single execution
#
# Usage: ./apply_topup_correction_multi_run.sh <bids_dir> <output_dir> <subject> <session> <task> [run1,run2,...]
#
# Example (single run): ./apply_topup_correction_multi_run.sh /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest run-01
# Example (multiple runs): ./apply_topup_correction_multi_run.sh /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest run-01,run-02,run-03
# Example (all runs): ./apply_topup_correction_multi_run.sh /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest all

set -e  # Exit on error

# Check for required arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <bids_dir> <output_dir> <subject> <session> <task> [run1,run2,... or 'all']"
    echo "Example (single run): $0 /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest run-01"
    echo "Example (multiple runs): $0 /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest run-01,run-02,run-03"
    echo "Example (all runs): $0 /data/bids /data/bids/derivatives/topup sub-01 ses-01 rest all"
    exit 1
fi

BIDS_DIR=$1
OUTPUT_DIR=$2
SUBJECT=$3
SESSION=$4
TASK=$5
RUN_SPEC=${6:-"all"}  # Default to "all" if not specified

# Construct paths
FUNC_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/func"
FMAP_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/fmap"
SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${SESSION}"

# Create output directories
mkdir -p "${SUBJECT_OUTPUT_DIR}"

echo "=========================================="
echo "FSL Topup Correction (Multi-Run)"
echo "Subject: ${SUBJECT}"
echo "Session: ${SESSION}"
echo "Task: ${TASK}"
echo "=========================================="

# Determine which runs to process
if [ "$RUN_SPEC" = "all" ]; then
    # Find all runs for this task
    BOLD_FILES=($(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_run-*_bold.nii*" | sort))
    
    if [ ${#BOLD_FILES[@]} -eq 0 ]; then
        # Try without run label (single run case)
        BOLD_FILES=($(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*bold.nii*" | grep -v "run-" | head -n 1))
    fi
    
    if [ ${#BOLD_FILES[@]} -eq 0 ]; then
        echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}_task-${TASK}"
        exit 1
    fi
    
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
else
    # Process specific runs
    IFS=',' read -ra RUNS <<< "$RUN_SPEC"
    BOLD_FILES=()
    for run in "${RUNS[@]}"; do
        run=$(echo "$run" | xargs)  # Trim whitespace
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

# Convert PE direction to FSL format (for acqparams.txt)
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
    
    # Find corresponding EPI and SBREF files
    if [ -n "$RUN_LABEL" ]; then
        EPI_FILE=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*epi.nii*" | head -n 1)
        SBREF_FILE=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_${RUN_LABEL}_*sbref.nii*" | head -n 1)
    else
        EPI_FILE=$(find "${FMAP_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*epi.nii*" | grep -v "run-" | head -n 1)
        SBREF_FILE=$(find "${FUNC_DIR}" -name "${SUBJECT}_${SESSION}_task-${TASK}_*sbref.nii*" | grep -v "run-" | head -n 1)
    fi
    
    # Check if EPI file exists
    if [ -z "$EPI_FILE" ]; then
        echo "Error: EPI file not found for this run"
        echo "Skipping this run..."
        continue
    fi
    
    echo "Found files:"
    echo "  BOLD: ${BOLD_FILE}"
    echo "  EPI: ${EPI_FILE}"
    if [ -n "$SBREF_FILE" ]; then
        echo "  SBREF: ${SBREF_FILE}"
    fi
    
    # Extract base filenames
    BOLD_BASE=$(basename "${BOLD_FILE}" .nii.gz)
    BOLD_BASE=$(basename "${BOLD_BASE}" .nii)
    EPI_BASE=$(basename "${EPI_FILE}" .nii.gz)
    EPI_BASE=$(basename "${EPI_BASE}" .nii)
    
    # Read phase encoding direction from JSON sidecar files
    BOLD_JSON="${BOLD_FILE%.nii*}.json"
    EPI_JSON="${EPI_FILE%.nii*}.json"
    
    if [ ! -f "$BOLD_JSON" ]; then
        echo "Error: JSON sidecar not found: ${BOLD_JSON}"
        echo "Skipping this run..."
        continue
    fi
    
    if [ ! -f "$EPI_JSON" ]; then
        echo "Error: JSON sidecar not found: ${EPI_JSON}"
        echo "Skipping this run..."
        continue
    fi
    
    # Extract phase encoding info from JSON
    BOLD_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${BOLD_JSON}" | cut -d'"' -f4)
    EPI_PE=$(grep -o '"PhaseEncodingDirection"[[:space:]]*:[[:space:]]*"[^"]*"' "${EPI_JSON}" | cut -d'"' -f4)
    BOLD_TRT=$(grep -o '"TotalReadoutTime"[[:space:]]*:[[:space:]]*[0-9.]*' "${BOLD_JSON}" | awk '{print $2}')
    EPI_TRT=$(grep -o '"TotalReadoutTime"[[:space:]]*:[[:space:]]*[0-9.]*' "${EPI_JSON}" | awk '{print $2}')
    
    echo ""
    echo "Phase encoding parameters:"
    echo "  BOLD PE direction: ${BOLD_PE}"
    echo "  EPI PE direction: ${EPI_PE}"
    echo "  BOLD Total Readout Time: ${BOLD_TRT}"
    echo "  EPI Total Readout Time: ${EPI_TRT}"
    
    BOLD_PE_VEC=$(convert_pe_to_vector "${BOLD_PE}")
    EPI_PE_VEC=$(convert_pe_to_vector "${EPI_PE}")
    
    # Step 1: Extract first volume from BOLD (or use SBREF if available)
    echo ""
    echo "Step 1: Preparing reference images..."
    
    if [ -n "$SBREF_FILE" ]; then
        echo "  Using SBREF as BOLD reference"
        BOLD_REF="${WORK_DIR}/bold_ref.nii.gz"
        cp "${SBREF_FILE}" "${BOLD_REF}"
    else
        echo "  Extracting first volume from BOLD"
        BOLD_REF="${WORK_DIR}/bold_ref.nii.gz"
        fslroi "${BOLD_FILE}" "${BOLD_REF}" 0 1
    fi
    
    # Extract first volume from EPI (topup scan)
    echo "  Extracting first volume from EPI"
    EPI_REF="${WORK_DIR}/epi_ref.nii.gz"
    fslroi "${EPI_FILE}" "${EPI_REF}" 0 1
    
    # Step 2: Merge the two images for topup
    echo ""
    echo "Step 2: Merging images for topup..."
    MERGED="${WORK_DIR}/merged_b0.nii.gz"
    fslmerge -t "${MERGED}" "${BOLD_REF}" "${EPI_REF}"
    
    # Step 3: Create acquisition parameters file
    echo ""
    echo "Step 3: Creating acquisition parameters file..."
    ACQPARAMS="${WORK_DIR}/acqparams.txt"
    echo "${BOLD_PE_VEC} ${BOLD_TRT}" > "${ACQPARAMS}"
    echo "${EPI_PE_VEC} ${EPI_TRT}" >> "${ACQPARAMS}"
    
    echo "  Acquisition parameters:"
    cat "${ACQPARAMS}"
    
    # Step 4: Run topup
    echo ""
    echo "Step 4: Running topup (this may take several minutes)..."
    TOPUP_BASENAME="${WORK_DIR}/topup_results"
    
    topup \
        --imain="${MERGED}" \
        --datain="${ACQPARAMS}" \
        --config=b02b0.cnf \
        --out="${TOPUP_BASENAME}" \
        --fout="${TOPUP_BASENAME}_field" \
        --iout="${TOPUP_BASENAME}_unwarped" \
        -v
    
    echo "  Topup completed successfully"
    
    # Step 5: Apply topup correction to BOLD data
    echo ""
    echo "Step 5: Applying topup correction to BOLD data..."
    
    CORRECTED_BOLD="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_bold.nii.gz"
    
    applytopup \
        --imain="${BOLD_FILE}" \
        --inindex=1 \
        --datain="${ACQPARAMS}" \
        --topup="${TOPUP_BASENAME}" \
        --out="${CORRECTED_BOLD}" \
        --method=jac \
        -v
    
    echo "  Applied topup to BOLD: ${CORRECTED_BOLD}"
    
    # Step 6: Copy important outputs to derivatives folder
    echo ""
    echo "Step 6: Organizing outputs..."
    
    # Copy the warp field (displacement field)
    WARP_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_warp.nii.gz"
    cp "${TOPUP_BASENAME}_field.nii.gz" "${WARP_OUTPUT}"
    
    # Copy the fieldmap coefficients (spline coefficients)
    COEF_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_fieldcoef.nii.gz"
    cp "${TOPUP_BASENAME}_fieldcoef.nii.gz" "${COEF_OUTPUT}"
    
    # Copy unwarped reference image
    UNWARP_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_unwarped.nii.gz"
    cp "${TOPUP_BASENAME}_unwarped.nii.gz" "${UNWARP_OUTPUT}"
    
    # IMPORTANT: Save the transformation matrix (movement parameters)
    # The topup results file contains the affine transformations
    MOVPAR_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_movpar.txt"
    if [ -f "${TOPUP_BASENAME}_movpar.txt" ]; then
        cp "${TOPUP_BASENAME}_movpar.txt" "${MOVPAR_OUTPUT}"
        echo "  Movement parameters: ${MOVPAR_OUTPUT}"
    else
        echo "  Warning: Movement parameters file not found (${TOPUP_BASENAME}_movpar.txt)"
    fi
    
    # Copy acquisition parameters for reference
    cp "${ACQPARAMS}" "${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_acqparams.txt"
    
    # Create a comprehensive JSON sidecar for the corrected BOLD
    CORRECTED_JSON="${CORRECTED_BOLD%.nii.gz}.json"
    cat > "${CORRECTED_JSON}" <<EOF
{
    "Description": "BOLD data corrected for susceptibility-induced distortions using FSL topup",
    "Sources": [
        "$(basename ${BOLD_FILE})",
        "$(basename ${EPI_FILE})"
    ],
    "TopupConfig": "b02b0.cnf",
    "ApplytopupMethod": "jac",
    "PhaseEncodingDirection": "${BOLD_PE}",
    "TotalReadoutTime": ${BOLD_TRT},
    "ProcessingOutputs": {
        "CorrectedBOLD": "$(basename ${CORRECTED_BOLD})",
        "WarpField": "$(basename ${WARP_OUTPUT})",
        "FieldCoefficients": "$(basename ${COEF_OUTPUT})",
        "UnwarpedReference": "$(basename ${UNWARP_OUTPUT})",
        "MovementParameters": "$(basename ${MOVPAR_OUTPUT})",
        "AcquisitionParameters": "$(basename ${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_acqparams.txt)"
    }
}
EOF
    
    echo ""
    echo "=========================================="
    echo "Run ${run_counter} completed successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  Corrected BOLD: ${CORRECTED_BOLD}"
    echo "  Warp field: ${WARP_OUTPUT}"
    echo "  Field coefficients: ${COEF_OUTPUT}"
    echo "  Unwarped reference: ${UNWARP_OUTPUT}"
    if [ -f "${MOVPAR_OUTPUT}" ]; then
        echo "  Movement parameters: ${MOVPAR_OUTPUT}"
    fi
    echo ""
    
    # Optional: Clean up working directory for this run
    # Uncomment the following line to remove intermediate files
    # rm -rf "${WORK_DIR}"
    
done

echo ""
echo "=========================================="
echo "All runs completed!"
echo "=========================================="
echo "Processed ${run_counter} run(s) successfully"
echo "Output directory: ${SUBJECT_OUTPUT_DIR}"
echo ""
echo "Done!"