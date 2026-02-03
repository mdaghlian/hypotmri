# sdc 
Apply SDC, using either fsl or afni

uses last 4 images of main bolds as "forward" - and "sdc" image as "backward" 


For fsl
```bash
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR="${PROJ_DIR}/derivatives/freesurfer"

bash s01_sdc.sh --bids_dir $BIDS_DIR --output_dir $BIDS_DIR/derivatives/sf1_sdcFSL --sub sub-hp01 --ses ses-01 --task pRFLE 
```

Or for afni
```bash
bash s01B_sdc_AFNI.sh --bids_dir $BIDS_DIR --output_dir $BIDS_DIR/derivatives/sf1_sdcAFNI --sub sub-hp01 --ses ses-01 --task pRFLE 
```