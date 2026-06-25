# Local setup

For HPC, see `hpc_setup.md` instead. This guide is for running the pipeline on your own machine.

## Before you start — requirements

Install these first (details for each are in the numbered sections below):

1. **Python environment manager** — mamba, micromamba, or conda
2. **Container manager** — Docker (local installs typically use this) or Singularity/Apptainer (typically used on HPC, see `hpc_setup.md`)
3. **FreeSurfer + freeview** — must be installed natively, not containerized, because you need the GUI to QC segmentations and make manual edits

## Step-by-step setup

### [0] Point your shell at this repo

`~/.bash_profile` is a file, stored in your home directory (on unix systems). You will not be able to see it because "." before file names makes them invisible. It runs automatically every time you open a new terminal - editing it is an easy way to ensure all the relevant and important paths / software is available. 

```bash
# Check whether ~/.bash_profile exists
ls ~/.bash_profile
# -> if you get the following it doesn't exist
# ls: cannot access ~/.bash_prfifds: No such file or directory
# Run 
touch ~/.bash_profile
```

Open `~/.bash_profile` in your text editor:

```bash
nano ~/.bash_profile
# or use your preferred editor (vi, gedit, code, etc.)
```

Add these two lines so every terminal knows where this repo lives and loads its settings:

```bash
# Inside ~/.bash_profile
export PIPELINE_DIR="/path/to/this/repository"
source "${PIPELINE_DIR}/config/config_pipeline.sh"
```

Then reload your current terminal:
```bash
source ~/.bash_profile
```

### [1] Create a project config file

Each dataset/project you run through this pipeline needs its own `project_<name>.sh` file living in `$PIPELINE_DIR/config/`. Copy this template, save it as `config/project_<yourproject>.sh`, and fill in your paths (**the part after `project_` in the filename must match `PROJ_NAME`**):

```bash
#!/bin/bash
# Save as: $PIPELINE_DIR/config/project_hypot.sh
export PROJ_NAME="hypot"
export BIDS_DIR="/path/to/your/bids/dataset"
export SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
export UCL_SERVER_ID="ucl-work"
export REMOTE_BIDS_DIR_PATH="/home/<your-hpc-username>/Scratch/projects/${PROJ_NAME}"
export REMOTE_BIDS_DIR="${UCL_SERVER_ID}:${REMOTE_BIDS_DIR_PATH}"
```

You can optionally add either of these to a project file to override the defaults set in `config_pipeline.sh`:
```bash
export PYPACKAGE_MANAGER="conda"        # default is "mamba"
export CONTAINER_TYPE="apptainer"       # default is "docker"; or "singularity"
```

`config/project_hypot.sh` and `config/project_stripe.sh` in this folder are working examples.

### [2] Activate a project

`config/project_current.sh` is a symlink pointing at whichever project file is "active". Set it with:

```bash
source set_project.sh hypot
```

This re-points the symlink to `project_hypot.sh` and re-sources `config_pipeline.sh`, so `$BIDS_DIR`, `$SUBJECTS_DIR`, etc. all update immediately. Switch projects at any time by re-running this with a different name — no need to restart your terminal.

**Verify it worked** by opening a *new* terminal and running:
```bash
source set_project.sh hypot
ls $BIDS_DIR        # should list your BIDS dataset
ls $SUBJECTS_DIR    # should list your FreeSurfer subjects directory
echo $PROJ_NAME     # should print "hypot"
```

If all three look right, move on to installation.

## Installing dependencies

Neuroimaging pipelines need a lot of software. To keep this manageable, everything is split between **conda/mamba environments** (Python packages) and **containers** (everything else — FreeSurfer/FSL/AFNI/fMRIPrep at pinned versions).

### [1] mamba or conda (required)

Lets you create an isolated Python install per pipeline stage, so packages with conflicting dependencies don't fight each other. Mamba does the same job as conda but resolves dependencies much faster — recommended unless you have a reason to prefer conda.

Install mamba: https://github.com/conda-forge/miniforge?tab=readme-ov-file#install

To use conda instead, set `PYPACKAGE_MANAGER="conda"` in your `project_*.sh` file (see [1] above).

### [2] Docker or Singularity/Apptainer (required)

Containers package an entire piece of software — OS, libraries, and all — so it runs identically regardless of your machine. We use this for the heavyweight, version-pinned tools (fMRIPrep, AFNI).

- **Docker** (recommended for local installs): https://docs.docker.com/desktop/setup/install/mac-install/. Docker Desktop must be running in the background for any `docker` command to work.
- **Singularity/Apptainer**: usually pre-installed on HPC clusters already — see `hpc_setup.md`. Only relevant locally if you specifically want to avoid Docker.

To use Singularity/Apptainer instead of Docker, set `CONTAINER_TYPE="apptainer"` (or `"singularity"`) in your `project_*.sh` file.

### [3] FreeSurfer + freeview (required)

FreeSurfer itself runs inside a container during the pipeline, but you still need a **native** install for `freeview` — the GUI used to QC segmentations and make manual edits.

1. Check the version this pipeline expects: `FREESURFER_VERSION` in `config/config_pipeline.sh` (currently `7.3.2`).
2. Download the build for your OS: https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/
3. Follow the install instructions: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
4. Get a free license key: https://surfer.nmr.mgh.harvard.edu/registration.html
5. Save the license file to **`$PIPELINE_DIR/config/license.txt`** — the pipeline reads `FSLICENSE` from this exact path.

### [4] MRI viewer (required)

You'll need to visually check registration, motion, and artefacts. **fsleyes** comes for free once the `preproc` conda environment is set up (see "Installing pipeline environments" below) — no separate install needed:

```bash
mamba activate preproc   # or: conda activate preproc
fsleyes
```

Any other viewer works too (e.g. ITK-SNAP) if you have a preference.

### [5] VS Code (optional, recommended)

Used for editing code and running notebooks. Any other editor/IDE works fine if you already have a preference.
- https://code.visualstudio.com/download

### [6] Everything else

Once [1]–[3] above are installed, the rest of the software stack is installed automatically by the scripts in the next section — conda/mamba creates the Python environments, and Docker/Singularity pulls the containers. If you are using Docker, make sure Docker Desktop is running before executing the ```s00_containers.sh``` command, or container pulls will fail.

## Installing pipeline environments & containers

Run this once your bash profile is set up (step [0] above) and you've opened a new terminal:

```bash
cd "${PIPELINE_DIR}/config"
python s00_python_environments.py   # creates the 5 conda/mamba environments
bash s00_containers.sh              # pulls/downloads the containers
```

This can take a while the first time (FreeSurfer/FSL packages and containers are large). Both scripts skip anything that already exists, so it's safe to re-run.

### What `s00_python_environments.py` creates

Five separate environments, one per pipeline stage, built from the recipes in `config/envs/*.yml`:

| Environment | Used by | Purpose |
|---|---|---|
| `preproc` | `functional/*` | FSL + nipype/nibabel/nilearn, used for distortion correction, coregistration, and confound extraction. Also gives you `fsleyes` (see [4] above) |
| `b14` | `anatomical/s02_b14atlas.py` | Benson14 visual area atlas (`neuropythy`) |
| `prf` | `postproc/*` | pRF/CF model fitting — `prfpy` (our fork), `dpu_mini`, and `cvl_utils` on top of numpy/scipy/scikit-learn |
| --- | --- | ---Optional may not include in final package --- |
| `autoflat` | `anatomical/s04_pycortex.py` | flattens FreeSurfer meshes for pycortex (`autoflatten`) |
| `pctx` | `anatomical/s04_pycortex.py`, postproc QC | pycortex surface visualisation — needs python 3.10 specifically, as required by pycortex |

`preproc` is built slightly differently from the rest: FSL doesn't publish a static yml, so the script downloads FSL's own conda recipe fresh each time (matched to your OS/CPU automatically) for the base install, then layers `preproc_extras.yml` on top — without pruning, so the FSL packages stay intact.

### What `s00_containers.sh` pulls

- **fMRIPrep** (`${FPREP_IMAGE}` for Docker / `${FPREP_SIF}` for Singularity) — used by the fMRIPrep stages in `anatomical/` and `functional/`
- **AFNI** (`${AFNI_IMAGE}` / `${AFNI_SIF}`) — used by `functional/s01_sdc_AFNI.py` for distortion correction
- **fsl_freesurfer** (`${FSL_FREESURFER_SIF}`, Singularity/Apptainer only) — a combined FSL+FreeSurfer image for HPC, where installing FreeSurfer natively isn't practical. **Not pulled when running locally with Docker** — locally you use your native FreeSurfer install ([3] above) plus the `preproc` environment's FSL ([4] above) instead.

On Singularity/Apptainer, all three `.sif` files are downloaded pre-built from Dropbox rather than built locally, to avoid a slow local Docker build. If you ever need to build one from scratch yourself, see `s00_containers_make_with_docker.sh`.

## Reinstalling / updating environments and containers

Re-running `s00_python_environments.py` with no flags is always safe — **existing environments are left untouched and skipped**. Use the flags below when you actually want to change something:

```bash
cd "${PIPELINE_DIR}/config"

# install whatever's missing (skips environments that already exist)
python s00_python_environments.py

# target a single environment instead of all 5
python s00_python_environments.py --env prf

# apply yml changes to an existing environment in place (non-destructive)
python s00_python_environments.py --env preproc --update

# wipe and fully recreate an environment from scratch
# (use after bumping a version pin in config_pipeline.sh or config/envs/*.yml)
python s00_python_environments.py --env prf --clean-env

# install into a custom location instead of the default conda envs dir (useful on HPC)
python s00_python_environments.py --env prf --prefix /scratch/$USER/envs

# create a parallel copy for testing, without touching your real environment
python s00_python_environments.py --env prf --env-suffix -test
```

`--clean-env` and `--update` are mutually exclusive — pick one.

To refresh containers:
```bash
bash s00_containers.sh
```
`docker pull` re-pulls the current tag for you automatically. For Singularity/Apptainer, the script always re-downloads each `.sif` from Dropbox — delete the existing `.sif` first if you want to guarantee a clean download rather than reusing whatever's cached locally.
