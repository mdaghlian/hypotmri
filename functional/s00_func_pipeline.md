# Functional pipeline 
First make sure you have the correct project activated 
```bash
source set_project.sh <project-name>
```

## [1] Susceptibility distortion correction - (afni)
run susceptibility distortion correction using AFNI. This corrects for the stretching/squishing of BOLD images, that occurs along the phase-encoding axis. AFNI does this by comparing an image acquired in one phase-encoding direction with another in the opposite direction. 

We call this command - specifying the subject, session and task to run through, and where to put the outputs. Note - you can also specify to run with the AFNI docker, by calling --afni-docker
#### Run SDC locally 
```bash
s01_sdc_AFNI.py --bids-dir $BIDS_DIR \
    --sub <sub-id> --ses <ses-id> \
    --task <task-id> \
    --output-file s1_AFNI_sdc \ 
    --afni-docker $AFNI_IMAGE
```
#### Run SDC on the HPC
```bash
s01_sdc_hpc.sh --bids-dir $BIDS_DIR \
    --sub <sub-id> --ses <ses-id> \
    --task <task-id> \
    --output-file s1_AFNI_sdc \ 

```


## [2] Motion correction (fsl)
Next, we can run motion correction for a whole session using fsl. Motion correction, coregistration uses the FreeSurfer T1, and surface projection, so freesurfer must have already been run before this step.

Coregistration strategy
-----------------------
1) Select first sbref as "master" - coregister to anatomy with bbregister
2) Coregister each sbref_i to sbref_master   (FLIRT, normcorr, DOF 6)
3) MCFLIRT per run, referencing the corresponding sbref_i
4) Concatenate transforms:  VOL -> sbref_i -> sbref_master -> FS_T1

```bash
$PYPACKAGE_MANAGER activate preproc
s02_coreg.py --sub hp01 --ses 01 \
    --input-dir $BIDS_DIR/s1_AFNI_sdc \
    --output-dir $BIDS_DIR/s2_coreg \
    --subjects-dir $SUBJECTS_DIR
```

For fsl
```bash 

python s01_sdc_fsl.py --bids-dir $BIDS_DIR --output-dir $BIDS_DIR/derivatives/sf1_sdc_fsl_test --sub sub-hp01 --ses ses-01 --task pRFLE 
```

For afni
```bash
python s01_sdc_AFNI.py --bids-dir $BIDS_DIR --output-dir $BIDS_DIR/derivatives/sf1_sdcAFNI_test --sub sub-hp01 --ses ses-01 --task pRFLE 
```