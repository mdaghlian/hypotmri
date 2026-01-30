#!/bin/bash
#
# OPTIMIZED: Apply Concatenated Transformations in Single Step
# Uses convertwarp to combine all FSL transforms before applying FreeSurfer transform
#
# Transform chain (applied in ONE resampling per volume):
# 1. Motion correction matrix (mcflirt) - per volume
# 2. Topup warp field - per run
# 3. Run alignment matrix (flirt) - per run
# 4. BBRegister to anatomical (FreeSurfer) - all runs
#
# Usage: ./s03_concat_transform.sh <preproc_dir> <alignment_dir> <output_dir> <subject_id>

set -e

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <preproc_dir> <alignment_dir> <output_dir> <subject_id>"
    echo ""
    echo "This script combines ALL transforms into the minimum number of interpolations."
    echo ""
    echo "Example:"
    echo "  $0 /data/derivatives/preproc/sub-01/ses-01/func \\"
    echo "     /data/derivatives/alignment/sub-01/ses-01 \\"
    echo "     /data/derivatives/final/sub-01/ses-01/func \\"
    echo "     sub-01"
    exit 1
fi

PREPROC_DIR=$1
ALIGN_DIR=$2
OUTPUT_DIR=$3
SUBJ_ID=$4
BIDS_FUNC_DIR=$5

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "OPTIMIZED Concatenated Transform Pipeline"
echo "Minimal interpolation strategy"
echo "Subject: $SUBJ_ID"
echo "=========================================="

# Check for required environment variables
if [ -z "$SUBJECTS_DIR" ]; then
    echo "Error: SUBJECTS_DIR not set. Please set FreeSurfer SUBJECTS_DIR"
    exit 1
fi

# Find all preprocessed BOLD runs
BOLD_FILES=($(find "$PREPROC_DIR" -name "*_desc-preproc_bold.nii.gz" | sort))

if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No preprocessed BOLD files found"
    exit 1
fi

echo "Found ${#BOLD_FILES[@]} run(s) to process"

# Get BBRegister transform
BBREG_LTA="$ALIGN_DIR/func_to_struct.lta"

if [ ! -f "$BBREG_LTA" ]; then
    echo "Error: BBRegister LTA not found: $BBREG_LTA"
    exit 1
fi

# Convert BBRegister LTA to FSL format for easier combination
echo "Converting BBRegister LTA to FSL format..."
BBREG_FSL="$OUTPUT_DIR/bbreg_func2anat.mat"

# Use lta_convert to get FSL format
# Note: This requires the reference functional and target anatomical
REF_FUNC="$ALIGN_DIR/grand_mean_all_runs.nii.gz"
TARGET_ANAT="$SUBJECTS_DIR/$SUBJ_ID/mri/brain.mgz"

if [ ! -f "$REF_FUNC" ]; then
    echo "Error: Reference functional not found: $REF_FUNC"
    exit 1
fi

# Convert FreeSurfer MGZ to NIFTI for FSL compatibility
TARGET_ANAT_NII="$OUTPUT_DIR/brain_anat.nii.gz"
mri_convert "$TARGET_ANAT" "$TARGET_ANAT_NII"

# Convert LTA to FSL matrix
# We'll use tkregister2 or lta_convert
lta_convert --inlta "$BBREG_LTA" --outfsl "$BBREG_FSL" \
            --src "$REF_FUNC" --trg "$TARGET_ANAT_NII"

echo "BBRegister transform converted to FSL format"

# Process each run
for BOLD_FILE in "${BOLD_FILES[@]}"; do
    
    BOLD_BASE=$(basename "$BOLD_FILE" .nii.gz)
    BOLD_BASE=${BOLD_BASE%_desc-preproc_bold}
    
    if [[ "$BOLD_BASE" =~ run-([0-9]+) ]]; then
        RUN_LABEL="run-${BASH_REMATCH[1]}"
    else
        RUN_LABEL=""
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: $BOLD_BASE"
    echo "=========================================="
    
    # Find transforms for this run
    TOPUP_WARP="$PREPROC_DIR/${BOLD_BASE}_desc-topup_warp.nii.gz"
    MCFLIRT_MATS="$PREPROC_DIR/${BOLD_BASE}_desc-mcflirt_transforms"
    
    if [ ! -f "$TOPUP_WARP" ] || [ ! -d "$MCFLIRT_MATS" ]; then
        echo "Warning: Required transforms not found, skipping..."
        continue
    fi
    
    # Find run-to-reference alignment
    UNWARPED_BASE="${BOLD_BASE}_desc-topup_unwarped"
    RUN_TO_REF_MAT="$ALIGN_DIR/${UNWARPED_BASE}_to_ref.mat"
    
    # Find original BOLD
    
    ORIGINAL_BOLD=$(find "$BIDS_FUNC_DIR" -name "${BOLD_BASE}.nii.gz" -o -name "${BOLD_BASE}.nii" | head -n 1)

    if [ -z "$ORIGINAL_BOLD" ]; then
        echo "Warning: Original BOLD not found, skipping..."
        continue
    fi
    
    NVOLS=$(fslnvols "$ORIGINAL_BOLD")
    echo "Processing $NVOLS volumes from: $(basename $ORIGINAL_BOLD)"
    
    # Create work directory
    WORK_DIR="$OUTPUT_DIR/work_${BOLD_BASE}"
    mkdir -p "$WORK_DIR"
    
    # ============================================================
    # OPTIMIZED APPROACH: Use convertwarp to combine FSL transforms
    # ============================================================
    
    echo ""
    echo "Creating combined warp fields..."
    
    # For each volume, create a combined warp that includes:
    # - Motion correction
    # - Topup
    # - Run alignment (if needed)
    # - BBRegister to anatomical
    
    TRANSFORMED_VOLS=()
    
    for (( vol=0; vol<$NVOLS; vol++ )); do
        
        if [ $((vol % 50)) -eq 0 ]; then
            echo "  Volume $vol/$NVOLS..."
        fi
        
        # Get motion correction matrix
        MAT_FILE=$(printf "$MCFLIRT_MATS/MAT_%04d" $vol)
        
        if [ ! -f "$MAT_FILE" ]; then
            echo "Warning: Motion matrix missing for volume $vol"
            continue
        fi
        
        # Combine affine matrices (motion + run-to-ref + bbreg)
        COMBINED_PREMAT="$WORK_DIR/premat_vol${vol}.mat"
        COMBINED_POSTMAT="$WORK_DIR/postmat_vol${vol}.mat"
        
        if [ -f "$RUN_TO_REF_MAT" ]; then
            # Chain: motion -> run-to-ref
            convert_xfm -omat "$COMBINED_PREMAT" \
                        -concat "$RUN_TO_REF_MAT" "$MAT_FILE"
        else
            # Just motion (this is the reference run)
            cp "$MAT_FILE" "$COMBINED_PREMAT"
        fi
        
        # The postmat is the BBRegister transform (func ref -> anat)
        cp "$BBREG_FSL" "$COMBINED_POSTMAT"
        
        # Create combined warp: combines topup warp + all affines
        COMBINED_WARP="$WORK_DIR/combined_warp_vol${vol}.nii.gz"
        
        # convertwarp combines warp fields and affine transforms
        # Order: apply premat, then warp, then postmat
        convertwarp \
            --ref="$TARGET_ANAT_NII" \
            --premat="$COMBINED_PREMAT" \
            --warp1="$TOPUP_WARP" \
            --postmat="$COMBINED_POSTMAT" \
            --out="$COMBINED_WARP"
        
        # Extract single volume from original data
        SINGLE_VOL="$WORK_DIR/vol_${vol}_orig.nii.gz"
        fslroi "$ORIGINAL_BOLD" "$SINGLE_VOL" $vol 1
        
        # Apply the combined warp in ONE step (single interpolation!)
        VOL_TRANSFORMED="$WORK_DIR/vol_${vol}_transformed.nii.gz"
        
        applywarp \
            --in="$SINGLE_VOL" \
            --ref="$TARGET_ANAT_NII" \
            --out="$VOL_TRANSFORMED" \
            --warp="$COMBINED_WARP" \
            --interp=spline
        
        TRANSFORMED_VOLS+=("$VOL_TRANSFORMED")
        
        # Clean up intermediate files to save space
        rm -f "$SINGLE_VOL" "$COMBINED_WARP" "$COMBINED_PREMAT"
        
    done
    
    echo ""
    echo "Merging transformed volumes..."
    
    # Create final 4D output
    FINAL_OUTPUT="$OUTPUT_DIR/${BOLD_BASE}_space-T1w_desc-preproc_bold.nii.gz"
    fslmerge -t "$FINAL_OUTPUT" "${TRANSFORMED_VOLS[@]}"
    
    echo "✓ Saved: $(basename $FINAL_OUTPUT)"
    
    # Create comprehensive JSON sidecar
    JSON_OUTPUT="${FINAL_OUTPUT%.nii.gz}.json"
    cat > "$JSON_OUTPUT" <<EOF
{
    "Description": "BOLD data in FreeSurfer anatomical space via concatenated transforms",
    "Space": "T1w",
    "SkullStripped": true,
    "Resolution": "Native anatomical",
    "ReferenceImage": "FreeSurfer brain.mgz for $SUBJ_ID",
    "TransformationPipeline": {
        "Step1": "Motion correction (mcflirt) - volume-specific affine",
        "Step2": "Susceptibility distortion correction (topup) - nonlinear warp",
        "Step3": "Run-to-reference alignment (flirt) - run-specific affine",
        "Step4": "Functional-to-anatomical registration (bbregister) - affine",
        "ConcatenationMethod": "FSL convertwarp",
        "NumberOfInterpolations": 1,
        "InterpolationMethod": "Spline"
    },
    "Sources": ["$(basename $ORIGINAL_BOLD)"],
    "NumberOfVolumes": $NVOLS,
    "ProcessingNotes": "All transforms combined using convertwarp and applied in single interpolation step to minimize blurring"
}
EOF
    
    # Clean up work directory (saves significant disk space)
    echo "Cleaning up temporary files..."
    rm -rf "$WORK_DIR"
    
    echo "✓ Complete: $BOLD_BASE"
    echo ""
    
done

echo "=========================================="
echo "SUCCESS: All runs transformed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Output directory: $OUTPUT_DIR"
echo "  - Space: FreeSurfer T1w (anatomical)"
echo "  - Interpolations per volume: 1 (OPTIMAL)"
echo "  - Transform chain: motion + topup + alignment + bbregister"
echo ""
echo "Quality notes:"
echo "  ✓ Minimal blurring (single interpolation)"
echo "  ✓ All spatial transforms concatenated"
echo "  ✓ Preserves maximum data quality"
echo ""
echo "Next steps:"
echo "  - Visual QC of alignment"
echo "  - Temporal filtering"
echo "  - Confound regression"
echo "  - Statistical analysis"
echo ""
echo "Done!"