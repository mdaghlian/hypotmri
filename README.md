# Pipeline for processing hypotony data

TODO:

[] setup+installation instructions + bidsifying 

[] In general improve the session name handling

[] Better docs; examples QC popups

[] Containerisation 


## setup
Requirements: 

Local:
- fsl 
- freeview + freesurfer
- 
### [1] Installation - TODO
### [2] BIDS(ish)ification - TODO

## anatomical
### [1] s01_fmriprep_anat_only
- Input: MP2RAGE T1w scan, in subject folder
- Output: Freesurfer surfaces + segmentations in fmriprep folder 
	- /BIDS/derivatives/fmriprep/sub-00/ses-fprep/anat
	- /BIDS/derivatives/freesurfer/sub-00
	- /BIDS/derivatives/freesurfer/sub-00_ses-fprep 
	- *note - fmriprep now forces freesurfer subject name to include session. To get around this we symlink a short subject name to the full one. Hence both files appear in directory*
### [2] s02_b14atlas
- Input: Subject freesurfer folder
- Output: Benson atlas ROI definitions & eccentricity + polar angle estimates (inside freesurfer folder)
### [3] Autoflatten (*may remove from final pipeline*)
- Input: Subject freesurfer folder
- Output: flattened cortical files
### [4] s04_pycortex (*may remove from final pipeline*)
- Input: Subject freesurfer folder
- Output: Pycortex files, for quick flattening & visualization
## functional

### [1] s01_sdc
- Input: 
	- "func" folder 
		- including "\_bold.nii", "\_sbref.nii"
	- "fmap" folder, including
		- topup/blip-down scans ("\_topup.nii")
- Intermediates:
	- Also includes calculated warps etc. and a bunch of other stuff. May remove later
- Output
	- "func_sdc" folder
	- susceptibility distortion corrected files sbref & bold

### [2] s02_bbreg_mcflirt
- Input:
	- "func_sdc" folder (output of [1]), including
		- "\_bold.nii", "\_sbref.nii"
- Intermediate files stored:
	- mcflirt within run coregistration matrix
	- run specific bref to bref master coregistration matrix
	- bref to anatomy coregistration matrix
	- bref to bref examples (used for QC)
- Output files
	- "func_coreg"
	- Volume-Anatomy per run per TR coregistration matrix
	- "\_preproc-bold" files (aligned & motion corrected volumes)
	- "\_space-fsnative" (aligned & projected surface files)
### [3] s03_confounds
- Input:
	- "func" directory: including "\_preproc-bold" files
- Intermediate: 
	- Fake "session" is created under subject directory
- Output:
	- fmriprep file, alongside the confounds we want
### [4] s04_denoising
TODO

## postproc

TODO

- prf analyses....


### experimental

scripts i'm messing with but haven't felt ready to delete or add to the main pipeline yet...