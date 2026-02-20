# Pipeline for processing hypotony data

TODO:

[] setup+installation instructions + bidsifying 

[] In general improve the session name handling

[] Better docs; examples QC popups

[] Containerisation 


## setup
Requirements: 
Local:

- docker or singularity
- fsl 
- freeview + freesurfer

add the following to your .bash_profile 
```bash
source "/path/to/this/folder/config/config_pipeline.sh"
```
Close and reopen terminal, this should be available then 

### [1] Downloads & Installation 
*will create option to work with singularity too*

Installing all the many, many, packages for neuroimaging analyses is a pain. Keeping it consistent, and reusable is even more of a pain. Fortunately we can use a couple of programs to manage the many packages and make installing a little easier

[1] mamba/conda (required)
This manages python environments. Basically, it allows you to create install a different python for each project. This is useful because different python programs have different requirements which often get in the way and cause havoc. Mamba is a speedy way to control this. You can also use conda, which will do the same things, but is a little slower. For installation instructions for mamba go to:
- https://github.com/conda-forge/miniforge?tab=readme-ov-file#install 


[2] Docker (required)
Docker allows you to make entire virtual machines, like a mini-version of a computer. You can create specific recipes for that machine, which contain all the software you need. Again this avoids trouble with different hardware, allows you to control versions etc. Follow these instructions to install docker:
- https://docs.docker.com/desktop/setup/install/mac-install/

[3] freesurfer+freeview (required)
The pipeline relies on freesurfer for segmentation; which needs to be checked by eye manually. This means it does need to be installed locally, as we need to be able to check surfaces & make edits with the gui. Check the freesurfer version in the config_pipeline.sh file. Currently we are working with 7.3.2. Select the correct version for your system (mac, linux, windows etc)
- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/
Then follow the instructions here
- https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
You will need to obtain a license key for your email
- https://surfer.nmr.mgh.harvard.edu/registration.html
Save this license file, inside the folder config, in this repository
- /where/you/cloned/this/repo/config/license.txt

[4] MRI viewer (optional)
It is also important to have a way to view the images, to check registration, motion, artefacts etc. Everyone will have there own preference. A standard one is to use **fsleyes**. If you don't have it installed already it can be installed with mamba. 
```bash
# We create a new environment with mamba, and install fsleyes to 
mamba create -n fslmamba -c conda-forge -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ fsl-base fsleyes
# 2. Activate the environment
conda activate fslmamba
```
I quite like itksnap; but anything you are used is good. 

[5] VS code(optional - highly recommended):
Vscode - use this to edit all of your programming files etc. It is also useful for running notebooks for code. It also can be used to install helpful plugins, like niivue. You could use another IDE if you are used to working in a specific environment. 
- https://code.visualstudio.com/download 
- TODO: add examples of notebooks

[6] Everything else...
Once you have this installed you are ready to go. All the other installation steps will be (hopefully!) handelled along the way by either *mamba* or *docker*. Remember for docker commands to work they need to be running  




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