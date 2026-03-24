#!/bin/bash

echo "Pulling relevant images using ${CONTAINER_TYPE}..."

case "${CONTAINER_TYPE}" in
    docker)
        # docker pull "${NEURODOCKER_IMAGE}"
        docker pull "${FPREP_IMAGE}"
        docker pull "${AFNI_IMAGE}"
        # docker pull "${NIGHRES_IMAGE}"
        ;;
    apptainer|singularity)
        # ${CONTAINER_TYPE} pull ${NEURODOCKER_SIF}  "docker://${NEURODOCKER_IMAGE}"
        ${CONTAINER_TYPE} pull ${FPREP_SIF}     "docker://${FPREP_IMAGE}"
        ${CONTAINER_TYPE} pull ${AFNI_SIF}      "docker://${AFNI_IMAGE}"
        # ${CONTAINER_TYPE} pull ${NIGHRES_SIF}   "docker://${NIGHRES_IMAGE}"
        ;;
    *)
        echo "Error: unknown CONTAINER_TYPE '${CONTAINER_TYPE}'. Set to 'docker', 'apptainer', or 'singularity'." >&2
        exit 1
        ;;
esac

echo "Done."