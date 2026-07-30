#!/bin/bash
# Functional alignment
# Use the unwarped, sbrefs & align them to each other using flirt
# Usage: ./s02_functional_alignment.sh <input_folder> <output_folder>

# Check if correct number of arguments are provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

IN_DIR=$1
OUT_DIR=$2
SUBJ_ID=$3

# Create output directory if it doesn't exist
mkdir -p "$OUT_DIR"

# Find files matching the pattern
# Using -f to get the full path
FILES=($(ls "$IN_DIR"/sub-*_ses-*_task-*_run-*_bold_desc-topup_unwarped_ref.nii.gz 2>/dev/null))

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No matching files found in $IN_DIR"
    exit 1
fi

echo "Found ${#FILES[@]} files. Processing..."

# 1. Copy over
processed_files=()
for i in "${!FILES[@]}"; do
    src_file="${FILES[$i]}"
    base_name=$(basename "$src_file" .nii.gz)
    fout="$OUT_DIR/$base_name.nii.gz"
    cp "$src_file" "$OUT_DIR/"
    
    # Store the path to the mean image for coregistration
    processed_files+=("$fout")
done

# 2. Coregister to the first image as reference
# Reference is the first 
REFERENCE="${processed_files[0]}"
REF_NAME=$(basename "$REFERENCE" .nii.gz)

echo "------------------------------------------------"
echo "Reference image for registration: $REF_NAME"
echo "------------------------------------------------"

registered_files=("$REFERENCE") # The first one is already 'registered' to itself

for (( i=1; i<${#processed_files[@]}; i++ )); do
    curr_file="${processed_files[$i]}"
    curr_base=$(basename "$curr_file" .nii.gz)
    reg_out="$OUT_DIR/${curr_base}_registered.nii.gz"
    mat_out="$OUT_DIR/${curr_base}_to_ref.mat"
    
    echo "Registering $curr_base to reference..."
    
    # flirt -in <input> -ref <reference> -out <output> -omat <matrix>
    # Using 6 degrees of freedom (rigid body) for same-subject intra-modal registration
    flirt -in "$curr_file" -ref "$REFERENCE" -out "$reg_out" -omat "$mat_out" -dof 6
    
    registered_files+=("$reg_out")
done

# 3. Calculate Grand Mean of all coregistered images
echo "Calculating grand mean..."
GRAND_MEAN_OUT="$OUT_DIR/grand_mean_all_runs.nii.gz"

# Merge all registered files into a 4D volume then take the mean across time/volumes
fslmerge -t "$OUT_DIR/tmp_merge.nii.gz" "${registered_files[@]}"
fslmaths "$OUT_DIR/tmp_merge.nii.gz" -Tmean "$GRAND_MEAN_OUT"

# Cleanup temporary merge file
rm "$OUT_DIR/tmp_merge.nii.gz"


# 4. Coregister Grand Mean to FreeSurfer Anatomy using bbregister
echo "------------------------------------------------"
echo "Running bbregister: Aligning Grand Mean to Surface"
echo "------------------------------------------------"

# Define registration output names
REG_DAT="$OUT_DIR/func_to_struct.dat"
REG_LTA="$OUT_DIR/func_to_struct.lta"
REG_MINC="$OUT_DIR/func_to_struct.mgz" # Optional output in FreeSurfer format

# bbregister command:
# --s: Subject ID in your SUBJECTS_DIR
# --mov: The moving volume (your grand mean)
# --reg: The output registration file (LTA/DAT format)
# --init-fsl: Use FSL's initialization method
bbregister \
    --s "$SUBJ_ID" \
    --mov "$GRAND_MEAN_OUT" \
    --reg "$REG_DAT" \
    --lta "$REG_LTA" \
    --bold \
    --init-fsl \
    --o "$OUT_DIR/grand_mean_in_struct_space.nii.gz"

echo "Registration matrix saved to: $REG_DAT"
echo "All steps complete for $SUBJ_ID."


# 5. Convert bbregister transform to an FSL-style FLIRT matrix
echo "------------------------------------------------"
echo "Converting bbregister transform to FSL matrix"
echo "------------------------------------------------"

# Define FreeSurfer target volume (structural)
FS_TARG="$SUBJECTS_DIR/$SUBJ_ID/mri/brain.mgz"   # or orig.mgz / T1.mgz depending on your preference

# Make a NIfTI version of the target for FSL bookkeeping (optional but nice to have)
STRUCT_NII="$OUT_DIR/struct_brain.nii.gz"
mri_convert "$FS_TARG" "$STRUCT_NII"

# Output FSL matrix (maps MOV -> TARG in FSL voxel coordinates)
FSL_MAT_OUT="$OUT_DIR/func_to_struct_fsl.mat"

tkregister2 \
  --mov "$GRAND_MEAN_OUT" \
  --targ "$FS_TARG" \
  --reg "$REG_DAT" \
  --fslregout "$FSL_MAT_OUT" \
  --noedit

echo "FSL-style matrix saved to: $FSL_MAT_OUT"
