# mcflirt 
fslreorient step is *** CRUCIAL ***
if it 
```bash
export SUBJECTS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/freesurfer
bash s02_bbreg_mcflirt.sh --input_dir ~/projects/dp-clean-link/240522NG/hypot/derivatives/sf1_topupAFNI --output_dir ~/projects/dp-clean-link/240522NG/hypot/derivatives/sf2_mcalignAFNI --sub sub-hp01 --ses ses-01 --task pRFRE 
```


```bash
export SUBJECTS_DIR=/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot/derivatives/freesurfer
bash s02_bbreg_mcflirt.sh --input_dir ~/projects/dp-clean-link/240522NG/hypot/sub-hp01/ses-01/func --output_dir ~/projects/dp-clean-link/240522NG/hypot/derivatives/sf2_mcalign_hires --sub sub-hp01 --ses ses-01 --task pRFRE 
```