#!/bin/bash
# Trailing slash on the source is required: copy the repo *contents* into ~/pipeline/
# (without it rsync creates ~/pipeline/<repo-name>/ and every *_hpc.sh breaks).
rsync -avz \
    "${PIPELINE_DIR%/}/" \
    "ucl-work:~/pipeline/" \
    --exclude-from="${PIPELINE_DIR%/}/config/hpc_helpers/.rsyncignore" \
    --exclude="*.nii.gz" --exclude="*.nii"
# Re-point project_current.sh on the cluster (the synced symlink points at a local path).
# On a first install ~/.bash_profile isn't set up yet, so set_project.sh may not be on PATH.
ssh ucl-work "source ~/.bash_profile 2>/dev/null; \
    if command -v set_project.sh >/dev/null; then set_project.sh ${PROJ_NAME}; \
    else echo 'set_project.sh not on PATH yet: finish hpc_setup.md section 6, then run: source set_project.sh ${PROJ_NAME}'; fi"