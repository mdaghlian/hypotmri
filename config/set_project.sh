#!/usr/bin/env bash
#---------------------------------------------------------------------------------------------------------
function Usage {
    cat <<USAGE
---------------------------------------------------------------------------------------------------
set_project
Switch active project by updating the symlink project_current.sh in \$PIPELINE_DIR/config/

Usage:
  set_project <project_name>

Example:
  set_project hypot
---------------------------------------------------------------------------------------------------
USAGE
    exit 1
}

if [[ $# -lt 1 ]] ; then
  Usage >&2
  exit 1
fi

ln -sf "${PIPELINE_DIR}/config/project_${1}.sh" "${PIPELINE_DIR}/config/project_current.sh"
echo "SETTING PROJECT TO ${1}"
source ${PIPELINE_DIR}/config/config_pipeline.sh
echo "BIDS_DIR=${BIDS_DIR}"
echo "SUBJECTS_DIR=${SUBJECTS_DIR}"