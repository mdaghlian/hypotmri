#!/usr/bin/env bash
#---------------------------------------------------------------------------------------------------------
# set_remote_paths for HPC: set BIDS_DIR to REMOTE_BIDS_DIR_PATH and SUBJECTS_DIR to REMOTE_BIDS_DIR_PATH/derivatives/freesurfer
if [[ "${PC_LOCATION}" != "local" ]]; then
  echo "On HPC - setting BIDS_DIR to REMOTE_BIDS_DIR"
  export BIDS_DIR="${REMOTE_BIDS_DIR_PATH}"
  export SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
  export CONTAINER_TYPE="apptainer"
  export PYPACKAGE_MANAGER="conda"
fi
echo "BIDS_DIR=${BIDS_DIR}"
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"