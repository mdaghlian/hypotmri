#!/bin/bash

# --- Default Values ---
chmod +x "${PIPELINE_DIR}/config"
chmod +x "${PIPELINE_DIR}/anatomical"
chmod +x "${PIPELINE_DIR}/functional"

export PC_LOCATION="local"
# --- Software versions ---
export FREESURFER_VERSION="7.3.2"
export FSLICENSE="${PIPELINE_DIR}/config/license.txt"
export FSL_VERSION="6.0.7.19"

# --- CONTAINER SETTINGS ---
# -> docker 
export CONTAINER_TYPE="docker" # or singularity -> replace "docker" with "singularity "
export DOCKFILE_DIR="${PIPELINE_DIR}/dockfiles"
if [ ! -d "$DOCKFILE_DIR" ]; then
    mkdir -p "$DOCKFILE_DIR"
fi

export NEURODOCKER_VERSION="2.1.1" 
export FPREP_VERSION="25.2.4"
export AFNI_IMAGE="vnmd/afni_26.0.07"
export NIGHRES_IMAGE=gianfrancof0/nighres:latest #vnmd/nighres_1.5.2"
export NEURODOCKER_IMAGE="repronim/neurodocker:${NEURODOCKER_VERSION}"
export FPREP_IMAGE="nipreps/fmriprep:${FPREP_VERSION}"

# -> Singularity / apptainer
export SIF_DIR="${HOME}/Scratch/sifs/"
[[ ! -d "$SIF_DIR" ]] && mkdir -p "$SIF_DIR"
export SIF_DIR_LOCAL="${PIPELINE_DIR}/siffiles"
[[ ! -d "$SIF_DIR_LOCAL" ]] && mkdir -p "$SIF_DIR_LOCAL"
export NEURODOCKER_SIF="${SIF_DIR}/neurodocker-${NEURODOCKER_VERSION}.sif"
export FPREP_SIF="${SIF_DIR}/fmriprep-${FPREP_VERSION}.sif"
export AFNI_SIF="${SIF_DIR}/afni-26.0.07.sif"
export NIGHRES_SIF="${SIF_DIR}/nighres-latest.sif"


# --- Conda environments ---
export PYPACKAGE_MANAGER="mamba"
export NEUROPYTHY_VERSION="0.12.16"
export AUTOFLAT_VERSION="1.0.6"
export NIBABEL_VERSION="5.3.3"
export PYTHON_FOR_PIPELINE="3.11"
export NIPYPE_VERSION="1.10.0"
export ANTS_VERSION="0.6.3"
export NILEARN_VERSION="0.13.1"
export PYCTX_VERSION="1.3.0"


# --- PROJECT INFORMATION ---
export PROJ_NAME="hypot"
export BIDS_DIR="/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot"
export SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
export UCL_SERVER_ID="ucl-work"
export REMOTE_BIDS_DIR_PATH="/home/ucjvmdd/Scratch/projects/${PROJ_NAME}"
export REMOTE_BIDS_DIR="${UCL_SERVER_ID}:${REMOTE_BIDS_DIR_PATH}"
