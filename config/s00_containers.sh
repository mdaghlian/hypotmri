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
        
        # TO make from scratch...
        # # ${CONTAINER_TYPE} pull ${NEURODOCKER_SIF}  "docker://${NEURODOCKER_IMAGE}"
        # ${CONTAINER_TYPE} pull ${FPREP_SIF}     "docker://${FPREP_IMAGE}"
        # ${CONTAINER_TYPE} pull ${AFNI_SIF}      "docker://${AFNI_IMAGE}"
        # # ${CONTAINER_TYPE} pull ${NIGHRES_SIF}   "docker://${NIGHRES_IMAGE}"
        
        # But we can just download them from dropbox
        curl -L -o ${FPREP_SIF} "https://www.dropbox.com/scl/fi/33ipqt6msvupl0p6opj6k/fmriprep-25.2.4.sif?rlkey=gyiqi5k4aom7iay7t3c1li8wr&st=c40hy2mb&dl=1"
        curl -L -o ${AFNI_SIF} "https://www.dropbox.com/scl/fi/2wswmq7q6953chghjva2c/afni-26.0.07.sif?rlkey=nh4qmvnrq0t2qxs38nefmih81&st=ijp0g5lp&dl=1"
        ;;
    *)
        echo "Error: unknown CONTAINER_TYPE '${CONTAINER_TYPE}'. Set to 'docker', 'apptainer', or 'singularity'." >&2
        exit 1
        ;;
esac

echo "Done."