#!/bin/bash
docker run --rm -it \
    --entrypoint /bin/bash \
    -e BIDS_DIR=/data/ \
    -e SUBJECTS_DIR=/data//derivatives/freesurfer \
    -e FS_LICENSE=/opt/freesurfer/license.txt \
    -v $BIDS_DIR:/data \
    -v $FSLICENSE:/opt/freesurfer/license.txt \
    $1 