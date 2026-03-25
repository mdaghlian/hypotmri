#!/bin/bash
rsync -avz \
    "${PIPELINE_DIR}" \
    "ucl-work:~/pipeline/" \
    --exclude-from="${PIPELINE_DIR}/config/hpc_helpers/.rsyncignore" \
    --exclude="*.nii.gz" --exclude="*.nii"
