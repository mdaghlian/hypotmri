# Pipeline for processing hypotony data

## setup - full details [./config/local_setup.md](./config/local_setup.md)
In brief: 
- Install requires a python package manager of your choice (mamba/conda) & container manager (docker/singularity or apptainer)
- Scripts in /config manage most of the software installation
- Other (soft) requirements are freesurfer - though you maybe able to also run this with a container
- I also suggest having your favourite MRI viewer (e.g., fsleyes) available, as well as an IDE for editing code + notebooks (e.g., vscode) 


## Pipeline overview

### [0] BIDS(ish)ification - TODO

## anatomical
### [1] s01_fmriprep_anat_only.sh
- Input: MPRAGE T1w scan, in subject folder
- Output: Freesurfer surfaces + anatomical outputs from fmriprep 
	- *note - fmriprep now forces freesurfer subject name to include session. To get around this we symlink a short subject name to the full one. Hence both files appear in directory*

### [2] s02_b14atlas.py
- Input: Subject freesurfer folder
- Output: Benson atlas ROI definitions & eccentricity + polar angle estimates (inside freesurfer folder)

### [3] Autoflatten (*optional - may remove*)
- Input: Subject freesurfer folder
- Output: flattened cortical files

### [4] Pycortex (*optional - may remove from final pipeline*)
- Input: Subject freesurfer folder
- Output: pycortex folder with flat maps & automatically drawn b14 ROIs

## functional

### [1] s01_sdc_AFNI.py
- Input: 
	- functional files *_bold; single band reference (if available) *_sbref; phase reversed epis   
- Output
	- susceptibility distortion corrected files sbref & bold

### [2] s02_coreg.py
Motion corection plus alignment: 
- Volume -> Bref_i -> Bref_main -> Freesurfer anatomy

Per run each volume, is aligned to the corresponding runs "bref" - this can either be the single band reference file, or will be take a single volume from the run. 

Each runs - "bref" is registered to a "bref_main" (i.e., the bref for the first run). 

The bref_main is registered to the anatomy (freesurfer)

Per volume each of these coregistration matrices is concatenated to put it all together in one single transform putting all of the volumes into the same space. In addition the data is sampled to the cortical surface. 

- Input:
	- bold files (sdc)
	- Freesurfer folder
- Output files
	- Volume->Anatomy per run per TR coregistration matrix
	- "\_preproc-bold" files (aligned & motion corrected volumes)
	- "\_space-fsnative" (aligned & projected surface files)

### [3] s03_fmriprep_func.py
To produce the fMRIprep confounds (acompcor etc) - we take the coregistered functional data, and run fMRIprep on it.  
- Input:
	- "func" directory: including "\_preproc-bold" files
	- Freesurfer folder + fMRIprep anatomy folder
- Output:
	- fmriprep file, alongside the confounds we want

### [4] s04_confounds.py
Take the confounds, from fMRIprep & derive the motion related confounds from s02_coreg matrices. Use them to denoise the functional data. 

TODO: test different implementations & versions 

- Input: 
	- fMRIprep functional files + confounds
	- coreg file -> including registration matrices & motion corrected bold
- Output:
	- Per run confound design matrix
	- Denoised surface & volume files

## postproc

### [1] s01_gauss_prfpy.py
Run prf analyses
- Input:
	- project_*.yml file specifying settings
	- project_*_dm.npy file, giving design matrix of prf stimulus
	- surface data (denoised), *.gii files 
- Output:
	- Gaussian prf fits

### [3] s03_cf_anaylsis
Run connective field analyses
- Input:
	- project_*.yml file specifying settings
	- surface data (denoised), *.gii files 
	- Freesurfer folder
- Output:
	- Connective field fits


### experimental

Scripts in process but not ready to be part of main pipeline