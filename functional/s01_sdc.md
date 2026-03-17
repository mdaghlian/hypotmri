# sdc 
Apply SDC, using either fsl or afni

uses last 4 images of main bolds as "forward" - and "sdc" image as "backward" 

```bash
conda activate preproc
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/
SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
```

For fsl
```bash 

python s01_sdc_fsl.py --bids_dir $BIDS_DIR --output_dir $BIDS_DIR/derivatives/sf1_sdc_fsl_test --sub sub-hp01 --ses ses-01 --task pRFLE 
```

For afni
```bash
python s01_sdc_AFNI.py --bids-dir $BIDS_DIR --output-dir $BIDS_DIR/derivatives/sf1_sdcAFNI_test --sub sub-hp01 --ses ses-01 --task pRFLE 
```