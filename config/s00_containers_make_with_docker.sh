#!/bin/bash

echo "Pulling relevant images using ${CONTAINER_TYPE}..."
docker run --privileged --rm \
    -v "${SIF_DIR_LOCAL}":/data \
    ghcr.io/apptainer/apptainer:latest \
    apptainer build /data/fmriprep-${FPREP_VERSION}.sif docker://${FPREP_IMAGE}

docker run --privileged --rm \
    -v "${SIF_DIR_LOCAL}":/data \
    ghcr.io/apptainer/apptainer:latest \
    apptainer build /data/afni-26.0.07.sif docker://${AFNI_IMAGE}

echo "Done."