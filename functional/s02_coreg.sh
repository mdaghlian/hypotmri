#!/bin/bash
set -e

# --- COREGISTRATION STRATEGY ---
# - input dir, contains sdc correcte sbrefs + bold 
# assumes 1 sbref per bold, taken at the same time, so they are 
# good references for motion correction
# (1) Select first sbref as "master" - coregister to anatomy with bbregister
# (2) MCFLIRT per run, using the corresponding sbref_i
# (3) Coregister each sbref_i to sbref_master
# (4) Concatenate transforms
# TODO: 
# option for missing sbrefs
# projfrac different 
# - Vol -> sbref_i -> sbref_master -> anatomy 

# --- Default Values ---
SESSION="ses-01"
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" &> /dev/null && pwd)

# --- Usage Function ---
usage() {
    echo "Usage: $0 --input_dir <path> --output_dir <path> --subject <ID> [options]"
    echo ""
    echo "Required Arguments:"
    echo "  --input_dir     Path to the bold + sbref files"
    echo "  --output_dir    Path to the output derivatives directory"
    echo "  --sub           Subject label (e.g., sub-01)"
    echo ""
    echo "Optional Arguments:"
    echo "  --help          Display this help message"
    exit 1
}

# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_dir)    INPUT_DIR="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --sub)          SUBJECT="$2"; shift 2 ;;
        --ses)          SESSION="$2"; shift 2 ;;
        --help)         usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# help function
get_labels() {
    local infile="$1"
    local run_label=""
    local TASK="task-unknown"

    # Extract run label
    if [[ "$infile" =~ run-([0-9]+) ]]; then
        run_label="run-${BASH_REMATCH[1]}"
    fi

    # Extract task label
    if [[ "$infile" =~ task-([a-zA-Z0-9]+) ]]; then
        TASK="task-${BASH_REMATCH[1]}"
    fi

    # Return values (space-separated)
    echo "$run_label $TASK"
}

# --- Validation ---
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$SUBJECT" ]]; then
    echo "Error: --INPUT_DIR, --output_dir, and --subject are required."
    echo "Run with --help for details."
    exit 1
fi

# --- Status Summary ---
echo "-------------------------------------------------------"
echo "Processing: Motion Correction + regisration "
echo "-------------------------------------------------------"
echo " Input:     $INPUT_DIR"
echo " Output:    $OUTPUT_DIR"
echo " Subject:   $SUBJECT"
echo " Session:   $SESSION"
echo "-------------------------------------------------------"


# Construct paths
SUBJECT_OUTPUT_DIR="${OUTPUT_DIR}/${SUBJECT}/${SESSION}"
mkdir -p "${SUBJECT_OUTPUT_DIR}"

# ************************************************
# ************************************************
# (1) CREATE BREF_MASTER & register to anatomy
echo "Creating master reference"
BREF_MASTER=$SUBJECT_OUTPUT_DIR/${SUBJECT}_${SESSION}_BREF_MASTER.nii.gz
if [[ ! -f "$BREF_MASTER" ]]; then
    # ************************************************
    # COPY SBREF AS MASTER BREF
    # ************************************************
    echo "COPYING FIRST SBREF FILE IN THIS SESSION"
    echo "THIS WILL BE THE BREF_MASTER" 
    
    # Select the first SBREF, we will use this as the target for all runs
    SBREFs=($(find "${INPUT_DIR}" -name "${SUBJECT}_${SESSION}_*sbref*.nii*"))
    # Select the first element (index 0)
    SBREF_m="${SBREFs[0]}"
    if [[ -z "$SBREF_m" ]]; then
        echo "No SBREF files found."
        exit 1
    else
        echo "Using SBREF: $SBREF_m"
    fi
    if [[ "$SBREF_m" == *.nii ]]; then
        BREF_MASTER_nii="$SUBJECT_OUTPUT_DIR/${SUBJECT}_${SESSION}_BREF_MASTER.nii"
        cp "${SBREF_m}" "${BREF_MASTER_nii}"
        gzip "${BREF_MASTER_nii}"
        # rm "${BREF_MASTER_nii}"
        echo "Compressed to: $FILE"
    else
        cp "${SBREF_m}" "${BREF_MASTER}"
    fi
    # Make a note of what was run
    mc_note=$SUBJECT_OUTPUT_DIR/reference_method_notes.txt
    echo "sbref - ${SBREF_m} was used as BREF_MASTER" >> $mc_note
    # Make sure header is nice
    fslreorient2std "$BREF_MASTER"
fi
echo ""
echo "=========================================="
echo "Running bbregister if not run already (BREF_MASTER -> FS T1)"
echo "=========================================="

FS_T1_MGZ="$SUBJECTS_DIR/$SUBJECT/mri/brain.mgz"
FS_T1_NII="${SUBJECT_OUTPUT_DIR}/${SUBJECT}_desc-fsbrain.nii.gz"
if [[ ! -f "$FS_T1_NII" ]]; then
    mri_convert "$FS_T1_MGZ" "$FS_T1_NII"
    fslreorient2std "$FS_T1_NII"
fi

BBREG_DAT="${SUBJECT_OUTPUT_DIR}/${SUBJECT}_${SESSION}_desc-sbref2fs_bbr.dat"
SBREF2FS_FSLMAT="${SUBJECT_OUTPUT_DIR}/${SUBJECT}_${SESSION}_desc-sbref2fs_bbr_fsl.mat"

if [[ ! -f "$BBREG_DAT" || ! -f "$SBREF2FS_FSLMAT" ]]; then
    # Initialise with flirt
    flirt \
        -in $BREF_MASTER \
        -ref $FS_T1_NII -dof 6 \
        -cost mutualinfo -omat $SUBJECT_OUTPUT_DIR/sbref_initial_reg.mat

    # Convert initial FLIRT output matrix to FreeSurfer format
    tkregister2 --s $SUBJECT --mov $BREF_MASTER \
                --targ $FS_T1_NII \
                --fsl $SUBJECT_OUTPUT_DIR/sbref_initial_reg.mat \
                --reg $SUBJECT_OUTPUT_DIR/sbref_initial_reg.dat \
                --noedit
   
    # Then use it to initialize bbregister
    bbregister \
        --s "$SUBJECT" \
        --mov "$BREF_MASTER" \
        --init-reg $SUBJECT_OUTPUT_DIR/sbref_initial_reg.dat \
        --reg "$BBREG_DAT" \
        --fslmat "$SBREF2FS_FSLMAT" \
        --bold
    flirt \
        -in "$BREF_MASTER" \
        -ref "$FS_T1_NII" \
        -applyxfm -init "$SBREF2FS_FSLMAT" \
        -out  "$SUBJECT_OUTPUT_DIR/${SUBJECT}_${SESSION}_BREF_MASTER_aligned"
fi

echo "bbregister outputs:"
echo "  dat:    $BBREG_DAT"
echo "  fslmat: $SBREF2FS_FSLMAT"
echo "  fs T1:  $FS_T1_NII"











# ************************************************
# ************************************************
# (2) sbref_i to sbref_master 
# Process each run
BREF_FILES=($(find "${INPUT_DIR}" -name "${SUBJECT}_${SESSION}*_sbref*.nii*" | sort))    
run_counter=0
for BREF_FILE in "${BREF_FILES[@]}"; do
    run_counter=$((run_counter + 1))
    read RUN_LABEL TASK <<< "$(get_labels "$BREF_FILE")"
    flirt \
        -in $BREF_FILE \
        -ref $BREF_MASTER -dof 6 \
        -cost normcorr \
        -omat "${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}_brefi_to_bref_master.mat" \
        -out "${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}_brefi_to_bref_master"
done







# ************************************************
# ************************************************
# (3) MCFLIRT & concatenate
# Find all the BOLD runs
BOLD_FILES=($(find "${INPUT_DIR}" -name "${SUBJECT}_${SESSION}*_bold*.nii*" | sort))    
if [ ${#BOLD_FILES[@]} -eq 0 ]; then
    echo "Error: No BOLD files found for ${SUBJECT}_${SESSION}"
    exit 1
else
    echo "Found ${#BOLD_FILES[@]} run(s) to process"
fi

run_counter=0
for BOLD_FILE in "${BOLD_FILES[@]}"; do
    run_counter=$((run_counter + 1))
    
    echo ""
    echo "=========================================="
    echo "Processing run ${run_counter}/${#BOLD_FILES[@]}"
    echo "=========================================="
    
    read RUN_LABEL TASK <<< "$(get_labels "$BOLD_FILE")"
    
    # corresponding sbref
    SBREF_i=$(find "${INPUT_DIR}" -name "${SUBJECT}_${SESSION}_${TASK}_${RUN_LABEL}*sbref*.nii*" | head -n 1)
    # Create work directory for this run
    if [ -n "$RUN_LABEL" ]; then
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}"
    else
        WORK_DIR="${SUBJECT_OUTPUT_DIR}/${TASK}"
    fi
    mkdir -p "${WORK_DIR}"
    
    # Extract base filenames
    BOLD_BASE=$(basename "${BOLD_FILE}" .nii.gz)
    BOLD_BASE=$(basename "${BOLD_BASE}" .nii)

    # ============================================================
    # MOTION CORRECTION (using mcflirt)
    # ============================================================
    echo ""
    echo "=========================================="
    echo "Motion Correction with MCFLIRT"
    echo "=========================================="
    
    MCFLIRT_OUTPUT_name="${WORK_DIR}/bold_mcf"
    MCFLIRT_OUTPUT="${WORK_DIR}/bold_mcf.nii.gz"
    MCFLIRT_PARAMS="${WORK_DIR}/bold_mcf.par"
    MCFLIRT_MATS_DIR="${WORK_DIR}/bold_mcf.mat"
    if [[ ! -d "${MCFLIRT_MATS_DIR}" ]]; then
        mcflirt -in "${BOLD_FILE}" \
                -reffile "${SBREF_i}" \
                -out "${MCFLIRT_OUTPUT_name}" \
                -mats \
                -plots \
                -report
        echo "Motion correction completed"
    else
        echo "${MCFLIRT_MATS_DIR} already exists, not recomputing"
    fi
    echo "  Motion parameters: ${MCFLIRT_PARAMS}"

    # Save motion parameters to output directory
    MOTION_PARAMS_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt_motion.par"
    cp "${MCFLIRT_PARAMS}" "${MOTION_PARAMS_OUTPUT}"

    # ============================================================
    # CONCATENATE: (VOL->SBREF) then (SBREF->FS_T1)  => (VOL->FS_T1)
    # Apply ONCE with applyxfm4D (single interpolation)
    # ============================================================
    echo ""
    echo "=========================================="
    echo "Concatenating transforms + single-step resample"
    echo "=========================================="

    # MCFLIRT_MATS_DIR="${WORK_DIR}/bold_mcf.mat"
    if [[ ! -d "${MCFLIRT_MATS_DIR}" ]]; then
        echo "Error: mcflirt mats dir not found: ${MCFLIRT_MATS_DIR}"
        exit 1
    fi

    COMBINED_MATS_DIR="${WORK_DIR}/bold2fs.mat"
    mkdir -p "${COMBINED_MATS_DIR}"

    # mcflirt mats are typically MAT_0000.. ; applyxfm4D with -fourdigit expects MAT_0000 naming
    sbref_i_to_sbref_m="${SUBJECT_OUTPUT_DIR}/${TASK}_${RUN_LABEL}_brefi_to_bref_master.mat"
    for M in "${MCFLIRT_MATS_DIR}"/MAT_*; do
        BN=$(basename "$M")
        
        # Step 1: Concatenate VOL->SBREF_i with SBREF_i->SBREF_MASTER
        # Result: VOL->SBREF
        TMP_MAT="${COMBINED_MATS_DIR}/tmp_${BN}"
        convert_xfm -omat "$TMP_MAT" -concat "${sbref_i_to_sbref_m}" "$M"
        
        # Step 2: Concatenate VOL->SBREF_ma with SBREF->FS
        # Result: VOL->FS (your final combined transform)
        convert_xfm -omat "${COMBINED_MATS_DIR}/${BN}" -concat "${SBREF2FS_FSLMAT}" "$TMP_MAT"
        
        # Clean up temporary file
        rm "$TMP_MAT"
    done

    # BIDS STYLE NAMING 
    BOLD_FS_OUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_space-fsT1_desc-moco_bbreg_bold.nii.gz"

    # Apply per-volume affines in one go
    # We want to keep the native resolution though...
    # take first volume
    # Extract first motion-corrected volume as reference
    RES_REF_CORRECT_HD="${WORK_DIR}/res_ref_correct_header.nii.gz"
    RES_REF="${WORK_DIR}/res_ref.nii.gz"
    fslroi "${BOLD_FILE}" "${RES_REF}" 0 1
    
    
    # ********************
    # get voxel size from BOLD -> TODO improve this reslicing here
    # -> don't like how hacky it is atm 
    vox=$(fslval "${RES_REF}" pixdim1)
    flirt -in "${FS_T1_NII}" -ref "${FS_T1_NII}" -applyisoxfm "${vox}" -out "${RES_REF_CORRECT_HD}"
    # resample the FS T1 to that voxel size (still FS space)
    # -> CORRECT HEADER
    
    if [ ! -f "${BOLD_FS_OUT}" ]; then
        
        # Everything in full massive high resolution
        # applyxfm4D "${BOLD_FILE}" "${FS_T1_NII}" "${BOLD_FS_OUT}" "${COMBINED_MATS_DIR}" -fourdigit -interp trilinear
        
        # Use native resolution?
        applyxfm4D "${BOLD_FILE}" "${RES_REF_CORRECT_HD}" "${BOLD_FS_OUT}" "${COMBINED_MATS_DIR}" -fourdigit -interp trilinear
   
         
    fi
    echo "Single-step output:"
    echo "  ${BOLD_FS_OUT}"

    # Also save transforms if you want them
    TRANSFORMS_OUTPUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_desc-mcflirt+bbreg_transforms"
    mkdir -p "${TRANSFORMS_OUTPUT}"
    cp -r "${COMBINED_MATS_DIR}"/* "${TRANSFORMS_OUTPUT}/"

    # ============================================================
    # PROJECT TO SURFACE (create surface timeseries)
    # ============================================================
    echo ""
    echo "=========================================="
    echo "Projecting to cortical surface (GIFTI format)"
    echo "=========================================="

    for HEMI in lh rh; do
        # Convert hemisphere labels for GIFTI (L/R instead of lh/rh)
        if [[ "${HEMI}" == "lh" ]]; then
            HEMI_GIFTI="L"
        else
            HEMI_GIFTI="R"
        fi
        
        SURF_OUT="${SUBJECT_OUTPUT_DIR}/${BOLD_BASE}_space-fsnative_hemi-${HEMI_GIFTI}_bold.func.gii"
        
        if [[ ! -f "${SURF_OUT}" ]]; then
            # TODO -> MOVE DIRECTLY FROM RAW -> SURFACE, using concatenated transforms
            # atm this adds another interpolation(?)
            mri_vol2surf \
                --mov "${BOLD_FS_OUT}" \
                --hemi "${HEMI}" \
                --projfrac-avg 0.2 0.8 0.1 \
                --o "${SURF_OUT}" \
                --trgsubject "${SUBJECT}" \
                --cortex --regheader $SUBJECT 
            # --projfrac 0.5  --surf-fwhm 3
            
            echo "Created surface timeseries: ${SURF_OUT}"
        else
            echo "Surface timeseries already exists: ${SURF_OUT}"
        fi
    done
    echo ""
    echo "=========================================="
    echo "Run ${run_counter} Completed Successfully!"
    echo "=========================================="

    # Optional: Clean up working directory
    # Uncomment to remove intermediate files
    # rm -rf "${WORK_DIR}"
    
done
