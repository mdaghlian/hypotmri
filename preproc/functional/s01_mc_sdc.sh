#!/bin/bash
#
# FSL Comprehensive Preprocessing Script with Proper Ordering
# Order: Motion Correction (mcflirt) → Topup (SDC) → Combined Application
# This follows fMRIPrep and other best-practice pipelines
#
# Usage: ./fmri_preproc_full.sh <bids_dir> <output_dir> <subject> <session> <task> [run1,run2,... or 'all']
#
# Example: ./fmri_preproc_full.sh /data/bids /data/bids/derivatives/preproc sub-01 ses-01 rest all

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
FMAP_DIR="${BIDS_DIR}/${SUBJECT}/${SESSION}/fmap"
SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${SESSION}"

# Create output directories
mkdir -p "${SUBJECT_OUTPUT_DIR}"

echo "=========================================="
echo "FSL Comprehensive Preprocessing Pipeline"
echo "Order: Motion Correction → Topup → Application"
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
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}"
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
    
    # Read phase encoding direction from JSON sidecar files
    BOLD_JSON="${BOLD_FILE%.nii*}.json"
    EPI_JSON="${EPI_FILE%.nii*}.json"
    
    if [ ! -f "$BOLD_JSON" ] || [ ! -f "$EPI_JSON" ]; then
        echo "Error: JSON sidecar(s) not found"
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
    
    # ============================================================
    # STEP 1: MOTION CORRECTION (using mcflirt)
    # ============================================================
    echo ""
    echo "=========================================="
    echo "STEP 1: Motion Correction with MCFLIRT"
    echo "=========================================="
    
    MCFLIRT_OUTPUT_name="${WORK_DIR}/bold_mcf"
    MCFLIRT_OUTPUT="${WORK_DIR}/bold_mcf.nii.gz"
    MCFLIRT_PARAMS="${WORK_DIR}/bold_mcf.par"
    
    # Determine reference volume for motion correction
    if [ -n "$SBREF_FILE" ]; then
        echo "Using SBREF as motion correction reference"
        mcflirt -in "${BOLD_FILE}" \
                -reffile "${SBREF_FILE}" \
                -out "${MCFLIRT_OUTPUT_name}" \
                -mats \
                -plots \
                -report
    else
        echo "Using median volume as motion correction reference"
        mcflirt -in "${BOLD_FILE}" \
                -out "${MCFLIRT_OUTPUT_name}" \
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
    
    # ============================================================
    # STEP 2: PREPARE REFERENCE IMAGES FOR TOPUP
    # ============================================================
    echo ""
    echo "=========================================="
    echo "STEP 2: Preparing Reference Images for Topup"
    echo "=========================================="
    
    # Extract mean or first volume from motion-corrected BOLD
    if [ -n "$SBREF_FILE" ]; then
        echo "  Using SBREF as BOLD reference"
        BOLD_REF="${WORK_DIR}/bold_ref.nii.gz"
        cp "${SBREF_FILE}" "${BOLD_REF}"
    else
        echo "  Computing temporal mean of motion-corrected BOLD"
        BOLD_REF="${WORK_DIR}/bold_ref.nii.gz"
        fslmaths "${MCFLIRT_OUTPUT}" -Tmean "${BOLD_REF}"
    fi
    
    # Extract first volume from EPI
    echo "  Extracting first volume from EPI"
    EPI_REF="${WORK_DIR}/epi_ref.nii.gz"
    fslroi "${EPI_FILE}" "${EPI_REF}" 0 1
    
    # ============================================================
    # STEP 3: RUN TOPUP FOR DISTORTION ESTIMATION
    # ============================================================
    echo ""
    echo "=========================================="
    echo "STEP 3: Running Topup (SDC)"
    echo "=========================================="
    
    # Merge reference images for topup
    MERGED="${WORK_DIR}/merged_b0.nii.gz"
    fslmerge -t "${MERGED}" "${BOLD_REF}" "${EPI_REF}"
    
    # Create acquisition parameters file
    ACQPARAMS="${WORK_DIR}/acqparams.txt"
    echo "${BOLD_PE_VEC} ${BOLD_TRT}" > "${ACQPARAMS}"
    echo "${EPI_PE_VEC} ${EPI_TRT}" >> "${ACQPARAMS}"
    
    echo "Acquisition parameters:"
    cat "${ACQPARAMS}"
    
    # Run topup
    echo ""
    echo "Running topup (this may take several minutes)..."
    TOPUP_BASENAME="${WORK_DIR}/topup_results"
    
    topup \
        --imain="${MERGED}" \
        --datain="${ACQPARAMS}" \
        --config=b02b0.cnf \
        --out="${TOPUP_BASENAME}" \
        --fout="${TOPUP_BASENAME}_field" \
        --iout="${TOPUP_BASENAME}_unwarped" \
        -v
    
    SHIFT_MAP="${WORK_DIR}/topup_shiftmap.nii.gz"
    # Make these extra versions - can be used later 
    # when concatenating all the transforms (fmriprep style)
    fugue \
        --loadfmap="${TOPUP_BASENAME}_field.nii.gz" \
        --dwell="${BOLD_TRT}" \
        --saveshift="${SHIFT_MAP}"
    TOPUP_WARP_3D="${WORK_DIR}/topup_warp_3vol.nii.gz"
    convertwarp \
        --ref="${BOLD_REF}" \
        --shiftmap="${SHIFT_MAP}" \
        --shiftdir="${BOLD_PE}" \
        --out="${TOPUP_WARP_3D}"
    echo "Topup completed successfully"
    
    # ============================================================
    # STEP 4: APPLY COMBINED CORRECTIONS
    # ============================================================
    echo ""
    echo "=========================================="
    echo "STEP 4: Applying Combined Corrections"
    echo "=========================================="
    
    # Apply topup to the MOTION-CORRECTED data
    # This preserves the motion correction while adding distortion correction
    CORRECTED_BOLD="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-preproc_bold.nii.gz"
    
    echo "Applying topup to motion-corrected BOLD..."
    applytopup \
        --imain="${MCFLIRT_OUTPUT}" \
        --inindex=1 \
        --datain="${ACQPARAMS}" \
        --topup="${TOPUP_BASENAME}" \
        --out="${CORRECTED_BOLD}" \
        --method=jac \
        -v
    
    echo "  Preprocessing complete: ${CORRECTED_BOLD}"
    
    # ============================================================
    # STEP 5: SAVE ALL OUTPUTS
    # ============================================================
    echo ""
    echo "=========================================="
    echo "STEP 5: Organizing Outputs"
    echo "=========================================="
    
    # Copy important outputs

    # [1] unwarped reference (only volume 1)
    UNWARP_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_unwarped_ref.nii.gz"
    fslroi "${TOPUP_BASENAME}_unwarped.nii.gz" "${UNWARP_OUTPUT}" 0 1 

    # [2] warp field    
    TOPUP_WARP_3D_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_warp.nii.gz"
    cp "${TOPUP_WARP_3D}" "${TOPUP_WARP_3D_OUTPUT}"

    # Save movement parameters if available
    MOVPAR_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_movpar.txt"
    if [ -f "${TOPUP_BASENAME}_movpar.txt" ]; then
        cp "${TOPUP_BASENAME}_movpar.txt" "${MOVPAR_OUTPUT}"
    fi
    
    # Copy transformation matrices from mcflirt
    MCFLIRT_MATS_DIR="${WORK_DIR}/bold_mcf.mat"
    if [ -d "${MCFLIRT_MATS_DIR}" ]; then
        TRANSFORMS_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt_transforms"
        mkdir -p "${TRANSFORMS_OUTPUT}"
        cp -r "${MCFLIRT_MATS_DIR}"/* "${TRANSFORMS_OUTPUT}/"
        echo "  Transformation matrices: ${TRANSFORMS_OUTPUT}/"
    fi
    
#     # Create comprehensive JSON sidecar
#     CORRECTED_JSON="${CORRECTED_BOLD%.nii.gz}.json"
#     cat > "${CORRECTED_JSON}" <<EOF
# {
#     "Description": "Preprocessed BOLD data with motion correction (mcflirt) and susceptibility distortion correction (topup)",
#     "Sources": [
#         "$(basename ${BOLD_FILE})",
#         "$(basename ${EPI_FILE})"
#     ],
#     "PreprocessingSteps": [
#         "MotionCorrection",
#         "SusceptibilityDistortionCorrection"
#     ],
#     "MotionCorrectionMethod": "FSL MCFLIRT",
#     "MotionCorrectionReference": "$([ -n "$SBREF_FILE" ] && echo "SBREF" || echo "median volume")",
#     "SDCMethod": "FSL topup",
#     "TopupConfig": "b02b0.cnf",
#     "ApplytopupMethod": "jac",
#     "PhaseEncodingDirection": "${BOLD_PE}",
#     "TotalReadoutTime": ${BOLD_TRT},
#     "Outputs": {
#         "PreprocessedBOLD": "$(basename ${CORRECTED_BOLD})",
#         "MotionParameters": "$(basename ${MOTION_PARAMS_OUTPUT})",
#         "TransformationMatrices": "$(basename ${TRANSFORMS_OUTPUT})/",
#         "WarpField": "$(basename ${WARP_OUTPUT})",
#         "FieldCoefficients": "$(basename ${COEF_OUTPUT})",
#         "UnwarpedReference": "$(basename ${UNWARP_OUTPUT})",
#         "TopupMovementParameters": "$([ -f "${MOVPAR_OUTPUT}" ] && basename ${MOVPAR_OUTPUT} || echo "N/A")",
#         "AcquisitionParameters": "$(basename ${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-topup_acqparams.txt)"
#     }
# }
# EOF
    
    echo ""
    echo "=========================================="
    echo "Run ${run_counter} Completed Successfully!"
    echo "=========================================="
    echo ""
    echo "Output files:"
    echo "  Preprocessed BOLD: ${CORRECTED_BOLD}"
    echo "  Motion parameters: ${MOTION_PARAMS_OUTPUT}"
    if [ -d "${TRANSFORMS_OUTPUT}" ]; then
        echo "  Transformation matrices: ${TRANSFORMS_OUTPUT}/"
    fi
    echo "  Topup warp field: ${WARP_OUTPUT}"
    echo "  Topup field coefficients: ${COEF_OUTPUT}"
    echo "  Unwarped reference: ${UNWARP_OUTPUT}"
    echo ""
    
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
echo ""
echo "Preprocessing summary:"
echo "  1. Motion correction (mcflirt)"
echo "  2. Susceptibility distortion correction (topup)"
echo "  3. Combined application with minimal interpolation"
echo ""
echo "Done!"