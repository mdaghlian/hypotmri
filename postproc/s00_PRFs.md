# Functional pipeline 
First make sure you have the correct project activated 
```bash
source set_project.sh <project-name>
```

Make sure you have the relevant files, saved inside the "postproc" folder here: 

- ```project_###.yml``` -> yaml file containing your settings for prf fitting. Important info includes screen dimensions, the TR of your sequence. See ```eg_prf_settings.yml```  for an example of the information needed
- ```project_###_dm.npy``` a n-pixel X n-pixel X n-timepoints array of your design matrix used in pRF fitting.   

To see a set by step walkthrough go to the "s01_gauss_prfpy.ipynb" notebook, and run through the stages. 
## [1] PRF fitting (gauss - prfpy)

```bash
conda activate prf

s01_gauss_prfpy.py \
    --bids-dir $BIDS_DIR \
    --sub <sub-id> \
    --ses <ses-id> \
    --task <task> \
    --input-file <desnoised-file-name> \
    --output-file <prf-file-name> \
    --project <project-id> --roi <roi-to-fit>

# e.g., 

s01_gauss_prfpy.py \
    --bids-dir $BIDS_DIR \
    --sub hp01 \
    --ses 01 \
    --task pRFLE \
    --input-file  s4_denoised \
    --output-file s5_gauss_prfpy \
    --project hypot --roi b14_V1.

```

## [2] Visualization
Go to ```s02_prf_plotting.ipynb``` notebook, and it will walk through plotting the different parameters