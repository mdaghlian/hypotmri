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

# --- Helpers ---
make_identity_mat() {
  local out=$1
  cat > "$out" << EOF
1 0 0 0
0 1 0 0
0 0 1 0
0 0 0 1
EOF
}

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
BOLD_FILES=($(find "$BIDS_FUNC_DIR" -name "*_bold.nii.gz" -o  -name "*_bold.nii" | sort))

if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No preprocessed BOLD files found"
    exit 1
fi

echo "Found ${#BOLD_FILES[@]} run(s) to process"

# Get final BBRegister transform (functional to anatomical)
BBREG_FSL="$ALIGN_DIR/func_to_struct_fsl.mat"

if [ ! -f "$BBREG_FSL" ]; then
    echo "Error: BBRegister FSL not found: $BBREG_FSL"
    exit 1
fi

# Convert FreeSurfer MGZ to NIFTI for FSL compatibility
TARGET_ANAT="$SUBJECTS_DIR/$SUBJ_ID/mri/brain.mgz"
TARGET_ANAT_NII="$OUTPUT_DIR/brain_anat.nii.gz"
mri_convert "$TARGET_ANAT" "$TARGET_ANAT_NII"

# Process each run
for BOLD_FILE in "${BOLD_FILES[@]}"; do
    
    BOLD_BASE=$(basename "${BOLD_FILE%.gz}" .nii)
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
        exit 1
    fi
    # Find run-to-reference alignment
    UNWARPED_BASE="${BOLD_BASE}_desc-topup_unwarped"
    RUN_TO_REF_MAT="$ALIGN_DIR/${UNWARPED_BASE}_ref_to_ref.mat"
    
    if [ ! -f "$RUN_TO_REF_MAT" ]; then
        echo "  Note: Run-to-reference matrix not found; assuming reference run."
        echo "  Creating identity matrix at: $RUN_TO_REF_MAT"
        mkdir -p "$ALIGN_DIR"
        make_identity_mat "$RUN_TO_REF_MAT"
    fi

    # Work directory
    WORK_DIR="$OUTPUT_DIR/${BOLD_BASE}"
    mkdir -p "$WORK_DIR"

    # Compose the static post-matrix: (BBREG * RUN_TO_REF)
    # NOTE: If your RUN_TO_REF_MAT direction is opposite, invert it and use the inverse instead.
    STATIC_POST="$WORK_DIR/run_to_anat_static_post.mat"
    convert_xfm -omat "$STATIC_POST" -concat "$BBREG_FSL" "$RUN_TO_REF_MAT"

    # Split 4D -> 3D volumes
    echo "Splitting 4D run into volumes..."
    # rm -f "$WORK_DIR"/vol_*.nii.gz "$WORK_DIR"/vol_warped_*.nii.gz
    # fslsplit "$BOLD_FILE" "$WORK_DIR/vol_" -t

    # Apply combined warp per volume (minimum interpolation with per-volume motion)
    echo "Applying combined transform per volume..."
    i=0
    shopt -s nullglob
    VOLS=("$WORK_DIR"/vol_*.nii.gz)
    shopt -u nullglob

    if [ "${#VOLS[@]}" -eq 0 ]; then
        echo "Error: fslsplit produced no volumes."
        exit 1
    fi

    for VOL in "${VOLS[@]}"; do
        MC_MAT=$(printf "%s/MAT_%04d" "$MCFLIRT_MATS_DIR" "$i")

        if [ ! -f "$MCFLIRT_MATS/$MC_MAT" ]; then
            echo "Error: Missing motion matrix for volume $i: $MC_MAT"
            exit 1
        fi

        # Build per-volume combined warp: premat (motion) + warp1 (topup) + postmat (static)
        # Important: convertwarp expects premat to be a *single* 4x4 matrix file (NOT a directory).
        convertwarp \
            --ref="$TARGET_ANAT_NII" \
            --premat="$MCFLIRT_MATS/$MC_MAT" \
            --warp1="$TOPUP_WARP" \
            --postmat="$STATIC_POST" \
            --out="$WORK_DIR/warp_$(printf "%04d" "$i")" >/dev/null
        
        applywarp \
            --ref="$TARGET_ANAT_NII" \
            --in="$VOL" \
            --warp="$WORK_DIR/warp_${i}.nii.gz" \
            --out="$WORK_DIR/vol_warped_$(printf "%04d" "$i").nii.gz" >/dev/null
        exit 1 
        i=$((i+1))
    done

    OUT_4D="$OUTPUT_DIR/${BOLD_BASE}_space-T1w_desc-concatWarp_bold.nii.gz"
    echo "Merging warped volumes -> $OUT_4D"
    fslmerge -t "$OUT_4D" "$WORK_DIR"/vol_warped_*.nii.gz

    echo "Done: $OUT_4D"

done

echo "=========================================="
echo "SUCCESS: All runs transformed!"
echo "=========================================="