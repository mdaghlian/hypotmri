# Functional pipeline 
First make sure you have the correct project activated 
```bash
source set_project.sh <project-name>
```

## [1] Susceptibility distortion correction - (afni)
run susceptibility distortion correction using AFNI. This corrects for the stretching/squishing of BOLD images, that occurs along the phase-encoding axis. AFNI does this by comparing an image acquired in one phase-encoding direction with another in the opposite direction. 

We call this command - specifying the subject, session and where to put the outputs. By default every `*bold*.nii*` in `<bids>/sub/ses/func/` is processed; restrict this with `--include-files`, `--include-patterns` or `--exclude-patterns` (task/run labels are read from the filenames). Note - you can also specify to run with the AFNI docker, by calling --afni-docker
#### Run SDC locally 
```bash
# Activate correct python env
$PYPACKAGE_MANAGER activate preproc
# Run SDC (using afni docker)
s01_sdc_AFNI.py --bids-dir $BIDS_DIR \
    --sub <sub-id> --ses <ses-id> \
    --output-file s1_AFNI_sdc \
    --afni-docker $AFNI_IMAGE \
    --include-patterns "*task-<task-id>_*bold*.nii*"   # optional
```
#### Run SDC on the HPC
```bash
# --task is optional; it is forwarded as --include-patterns "*task-<task-id>_*bold*.nii*"
s01_sdc_hpc.sh --bids-dir $BIDS_DIR \
    --sub <sub-id> --ses <ses-id> \
    --task <task-id> \
    --output-file s1_AFNI_sdc
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
    --input-file s1_AFNI_sdc \
    --output-file s2_coreg \
    --bids-dir $BIDS_DIR
```
On the cluster
```bash
s02_coreg_hpc.sh --sub hp01 --ses 01 \
    --input-file s1_AFNI_sdc \
    --output-file s2_coreg \
    --bids-dir $BIDS_DIR
```
You can also submit runs & tasks separately - if you want to break it into smaller stages up. Select runs with `--include-files` / `--include-patterns` (globs relative to the input `func/` dir) and `--exclude-patterns`.
```bash
# Local
s02_coreg.py --sub hp01 --ses 01 \
    --input-file s1_AFNI_sdc \
    --output-file s2_coreg \
    --bids-dir $BIDS_DIR \
    --include-patterns "*task-pRFLE_run-01_*bold*.nii*"

# Cluster
# Arguments after " -- " are passed to s02_coreg.py
s02_coreg_hpc.sh --sub hp01 --ses 01 \
    --input-file s1_AFNI_sdc \
    --output-file s2_coreg \
    --bids-dir $BIDS_DIR \
    -- --include-patterns "*task-pRFLE_run-01_*bold*.nii*"


# you can loop over a submmission job like this
for TASK in "CSFLE" "CSFRE" "pRFLE" "pRFRE"; do
    for run in "01" "02" "03"; do
        s02_coreg_hpc.sh --sub hp01 --ses 01 \
            --input-file s1_AFNI_sdc \
            --output-file s2_coreg \
            --bids-dir $BIDS_DIR \
            --skip-sync \
            -- --include-patterns "*task-${TASK}_run-${run}_*bold*.nii*"
    done
done
```

## [3] Confounds (aCompCor / tCompCor / cosine)
There are two ways to make the CompCor confounds. **Option A (`s03_make_confounds.py`) is the recommended one.**

### [3A] s03_make_confounds.py — recommended
Reproduces fMRIPrep's aCompCor/tCompCor (same masks, parameters and column names) directly from the `s02_coreg` BOLD and the fMRIPrep-anat tissue maps, without running the fMRIPrep BOLD workflow. Much faster, and avoids re-resampling data that is already coregistered.

Writes `<sub>/<ses>/*desc-confounds_timeseries.tsv` per run, plus a subject-level BOLD-grid brain mask (`<sub>/<sub>_desc-acompcorBrain_mask.nii.gz`).

Locally
```bash
$PYPACKAGE_MANAGER activate preproc
s03_make_confounds.py --sub hp01 --ses 01 \
    --bids-dir $BIDS_DIR \
    --coreg-file s2_coreg \
    --anat-file fmriprep \
    --output-file s3_make_confounds
```
On the cluster
```bash
s03_make_confounds_hpc.sh --sub hp01 --ses 01 \
    --bids-dir $BIDS_DIR \
    --coreg-file s2_coreg \
    --anat-file fmriprep \
    --output-file s3_make_confounds
```

### [3B] s03_fmriprep_func.py — previous default
Runs the full fMRIPrep BOLD workflow on the coregistered data, only to get its confounds + brain mask. Writes to `derivatives/fmriprep`.

Locally 
```bash
$PYPACKAGE_MANAGER activate preproc
s03_fmriprep_func.py --sub hp01 --ses 01 \
    --input-file s2_coreg --bids-dir $BIDS_DIR
```
On the cluster
```bash
s03_fmriprep_func_hpc.sh --sub hp01 --ses 01 \
    --input-file s2_coreg --bids-dir $BIDS_DIR
```

## [4] confounds 

> **Which s04 reads which s03?** `s04_confounds.py` (PCA) only reads confounds from `derivatives/fmriprep` (i.e. [3B]). To use the recommended [3A] output, run `s04_confounds_nopca.py --confounds-file s3_make_confounds` (no HPC twin yet):
> ```bash
> s04_confounds_nopca.py --sub hp01 --ses 01 \
>     --bids-dir $BIDS_DIR \
>     --moco-file s2_coreg \
>     --confounds-file s3_make_confounds \
>     --output-file s4_denoised_nopca
> ```
> PCA vs no-PCA denoising is still an open decision (see `todo.md`).

`s04_confounds.py`:
- Grabs the confounds & bold-brainmasks from fmriprep
- Removes the motion related ones 
- Replace them with our own, FSL derived motion confounds (from moco file)
- Save as .tsv file 
- Run PCA on select components of confound file, apply sg-filter, save as .tsv 
- Volume data: Apply sg-filter & regress out PCA noise components
- Surface data: Apply sg-filter & regress out PCA noise components

Locally: 
```bash
s04_confounds.py --sub hp01 --ses 01 \
    --bids-dir $BIDS_DIR \
    --moco-file s2_coreg \
    --output-file s4_denoised
```

On the cluster:

```
s04_confounds_hpc.sh --sub hp01 --ses 01 \
    --bids-dir $BIDS_DIR \
    --moco-file s2_coreg \
    --output-file s4_denoised
```