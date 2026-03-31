#!/bin/bash

echo "Pulling relevant images using ${CONTAINER_TYPE}..."
docker run --privileged --rm \
    -v "${SIF_DIR_LOCAL}":/data \
    ghcr.io/apptainer/apptainer:latest \
    apptainer build /data/${FPREP_SIF} docker://${FPREP_IMAGE}

docker run --privileged --rm \
    -v "${SIF_DIR_LOCAL}":/data \
    ghcr.io/apptainer/apptainer:latest \
    apptainer build /data/${AFNI_SIF} docker://${AFNI_IMAGE}
    

# Combined FSL + FREESURFER 
# [1] create the dockfile 
docker run --rm "${NEURODOCKER_IMAGE}" \
    generate docker --pkg-manager apt \
    --base-image debian:bullseye-slim \
    --yes \
    --freesurfer version=$FREESURFER_VERSION \
    --fsl version=$FSL_VERSION \
    > "${DOCKFILE_DIR}/fsl_freesurfer.dockerfile"
# [2] build the docker image locally from the generated Dockerfile
docker build \
    --platform linux/amd64 \
    -f "${DOCKFILE_DIR}/fsl_freesurfer.dockerfile" \
    -t "${FSL_FREESURFER_IMAGE}" \
    "${PIPELINE_DIR}"  

# [3] Save Docker image to tar
docker save "${FSL_FREESURFER_IMAGE}" \
    -o "${SIF_DIR_LOCAL}/fsl_freesurfer.tar"

# [4] Convert tar → .sif via Apptainer
docker run --privileged --rm \
    -v "${SIF_DIR_LOCAL}":/data \
    ghcr.io/apptainer/apptainer:latest \
    apptainer build \
        /data/${FSL_FREESURFER_SIF} \
        docker-archive:///data/fsl_freesurfer.tar

# [5] Cleanup tar (optional)
rm "${SIF_DIR_LOCAL}/fsl_freesurfer.tar"