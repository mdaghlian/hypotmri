#!/bin/bash
dock_cmd=(
    docker run --rm "${NEURODOCKER_IMAGE}"
    generate "${CONTAINER_TYPE}"
        --pkg-manager apt
        --base-image debian:bullseye-slim
        --yes
        --miniconda
            version=latest
            env_name=b14
            conda_install="python=3.11"
            pip_install="nibabel neuropythy==0.12.16"
        --miniconda
            version=latest
            env_name=autoflat
            conda_install="python=3.11"
            pip_install="autoflatten==1.0.6"
)

# --fsl version=6.0.7.1
# --afni method=source version=latest 
#     conda_install="python=3.11 nibabel" 
#     pip_install="neuropythy=0.12.16"
# --afni method=binaries version=latest 

dock_id="a03"
echo "${dock_cmd[@]}"
rm -f "${DOCKFILE_DIR}"/*
"${dock_cmd[@]}" > "${DOCKFILE_DIR}/${dock_id}.Dockerfile"
docker build  -t hypot-${dock_id} -f "${DOCKFILE_DIR}/${dock_id}.Dockerfile" .
# --platform linux/amd64
# https://repronim.org/neurodocker/user_guide/quickstart.html
# --fsl version=${FSL_VERSION} \
# --afni method=binaries version=latest \


# neurodocker generate docker \
#         --pkg-manager apt \
#         --base-image debian:bullseye-slim \
#         --yes \
#         --ants version=2.4.3 \
#         --fsl version=6.0.7.1 \
#         --convert3d version=1.0.0 \
#         --install gcc g++ graphviz tree git-annex vim emacs-nox nano less ncdu tig octave netbase \
#         --miniconda \
#                 version=latest \
#                 mamba=true \
#                 conda_install="python=3.11 nipype pybids=0.16.3 pytest jupyterlab jupyter_contrib_nbextensions traits scikit-image seaborn nbformat nb_conda" \
#                 pip_install="nilearn=0.10.1 datalad[full] nipy duecredit nbval" \
#         --run 'jupyter nbextension enable exercise2/main && jupyter nbextension enable spellchecker/main' \
#         --run 'mkdir /data && chmod 777 /data && chmod a+s /data' \
#         --run 'mkdir /output && chmod 777 /output && chmod a+s /output' \
#         --spm12 version=r7771 \
#         --user neuro \
#         --run-bash 'cd /data
#                 && datalad install -r ///workshops/nih-2017/ds000114
#                 && cd ds000114
#                 && datalad update -r
#                 && datalad get -r sub-01/ses-test/anat sub-01/ses-test/func/*fingerfootlips*' \
#         --run 'curl -fL https://files.osf.io/v1/resources/fvuh8/providers/osfstorage/580705089ad5a101f17944a9 -o /data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c.tar.gz
#                 && tar xf /data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c.tar.gz -C /data/ds000114/derivatives/fmriprep/.
#                 && rm /data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c.tar.gz
#                 && find /data/ds000114/derivatives/fmriprep/mni_icbm152_nlin_asym_09c -type f -not -name ?mm_T1.nii.gz -not -name ?mm_brainmask.nii.gz -not -name ?mm_tpm*.nii.gz -delete' \
#         --copy . "/home/neuro/nipype_tutorial" \
#         --user root \
#         --run 'chown -R neuro /home/neuro/nipype_tutorial' \
#         --run 'rm -rf /opt/conda/pkgs/*' \
#         --user neuro \
#         --run 'mkdir -p ~/.jupyter && echo c.NotebookApp.ip = \"0.0.0.0\" > ~/.jupyter/jupyter_notebook_config.py' \
#         --workdir /home/neuro/nipype_tutorial \
#         --entrypoint jupyter-notebook \
# > nipype-tutorial.Dockerfile


# --miniconda create_env= \
#             conda_install="python=3.6 traits" \
#             pip_install="nipype"