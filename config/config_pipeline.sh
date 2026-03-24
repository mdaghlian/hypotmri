#!/bin/bash

# --- Default Values ---
chmod +x "${PIPELINE_DIR}/config"
chmod +x "${PIPELINE_DIR}/anatomical"
chmod +x "${PIPELINE_DIR}/functional"

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
export NEURODOCKER_SIF="neurodocker-${NEURODOCKER_VERSION}.sif"
export FPREP_SIF="fmriprep-${FPREP_VERSION}.sif"
export AFNI_SIF="afni-26.0.07.sif"
export NIGHRES_SIF="nighres-latest.sif"


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


