# Align & motion correct 
Takes in the susceptibility distortion corrected SBREF & BOLD runs from previous step 

Per run: 
- Motion correct volume -> bref_i (run specific bref)
- bref_i -> bref_master (first bref, selected as master)
- bref_master -> anatomy (via bbregister)
- Concatenate each of the transforms, per volume
- Output motion corrected, aligned volumes in anatomical space (but maintain original functional resolution)
- Output surface time series .gii, per hemisphere

```bash
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot
export SUBJECTS_DIR=${BIDS_DIR}/derivatives/freesurfer

bash s02_coreg.sh --input_dir $BIDS_DIR/derivatives/sf1_sdcFSL --output_dir $BIDS_DIR/derivatives/sf2_coregFSL9dof --sub sub-hp01 --ses ses-01 --bref_dof 9
```


```bash
BIDS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot
export SUBJECTS_DIR=${BIDS_DIR}/derivatives/freesurfer

bash s02_coreg_moco2master.sh --input_dir $BIDS_DIR/derivatives/sf1_sdcAFNI --output_dir $BIDS_DIR/derivatives/sf2_coregAFNImoco2master --sub sub-hp01 --ses ses-01 
```