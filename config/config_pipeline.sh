#!/bin/bash

# --- Default Values ---
export PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chmod +x "${PIPELINE_DIR}/config"
chmod +x "${PIPELINE_DIR}/anatomical"
chmod +x "${PIPELINE_DIR}/functional"

# Add to path
# export PATH="${PIPELINE_DIR}/config:${PIPELINE_DIR}/anatomical:${PIPELINE_DIR}/functional:$PATH"
# Settings & version control
export DOCKFILE_DIR="${PIPELINE_DIR}/dockfiles"
if [ ! -d "$DOCKFILE_DIR" ]; then
    mkdir -p "$DOCKFILE_DIR"
fi

export CONTAINER_TYPE="docker" # or singularity -> replace "docker" with "singularity "
export NEURODOCKER_VERSION="2.1.1" 
export FPREP_VERSION="25.2.4"
export FREESURFER_VERSION="7.3.2"
export FSLICENSE="${PIPELINE_DIR}/config/license.txt"
export FSL_VERSION="6.0.7.19"
export AFNI_VERSION="AFNI_26.0.07"

if [[ "$CONTAINER_TYPE" == "docker" ]]; then
    export NEURODOCKER_IMAGE="repronim/neurodocker:${NEURODOCKER_VERSION}"
    export FPREP_IMAGE="poldracklab/fmriprep:${FPREP_VERSION}"
elif [[ "$CONTAINER_TYPE" == "singularity" ]]; then
    echo TODO
else
    echo "Invalid CONTAINER_TYPE: $CONTAINER_TYPE. Must be 'docker' or 'singularity'."
    exit 1
fi

# Mamba environments
export NEUROPYTHY_VERSION="0.12.16"
export AUTOFLAT_VERSION="1.0.6"
export NIBABEL_VERSION="5.3.3"
export PYTHON_FOR_PIPELINE="3.11"


