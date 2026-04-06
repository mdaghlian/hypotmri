# Config - for LOCAL installation (for HPC see hpc_setup.md)
Requirements: 
Local:

- Container manager: **docker/singularity/apptainer**
- Python envirnoment manager: **mamba/conda**
- **freeview + freesurfer**

add the following to your .bash_profile 
```bash
export PIPELINE_DIR="/path/to/this/repository"
source "${PIPELINE_DIR}/config/config_pipeline.sh"
```
Next for each project you want to run with this pipeline, create a ```project_*.sh``` file inside ```$PIPELINE_DIR/config/``` (this folder). These files must have the following paths:
```bash
#!/bin/bash
# Put inside $PIPELINE_DIR/config/project_hypot.sh
# Change all the relevant paths...
# --- PROJECT INFORMATION ---
export PROJ_NAME="hypot"
export BIDS_DIR="/Users/marcusdaghlian/projects/dp-clean-link/240522NG/hypot"
export SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
export UCL_SERVER_ID="ucl-work"
export REMOTE_BIDS_DIR_PATH="/home/ucjvmdd/Scratch/projects/${PROJ_NAME}"
export REMOTE_BIDS_DIR="${UCL_SERVER_ID}:${REMOTE_BIDS_DIR_PATH}"
```
Here is another example for anothe project I'm running:
```bash
#!/bin/bash
# Put inside $PIPELINE_DIR/config/project_stripe.sh
# --- PROJECT INFORMATION ---
export PROJ_NAME="stripe"
export BIDS_DIR="/Users/marcusdaghlian/projects/dp-clean-link/pilot-clean/"
export SUBJECTS_DIR="${BIDS_DIR}/derivatives/freesurfer"
export UCL_SERVER_ID="ucl-work"
export REMOTE_BIDS_DIR_PATH="/home/ucjvmdd/Scratch/projects/${PROJ_NAME}"
export REMOTE_BIDS_DIR="${UCL_SERVER_ID}:${REMOTE_BIDS_DIR_PATH}"
```
The "current" project is set by the contents of "project_current.sh". You can quickly switch between projects by running: ```source set_project.sh hypot```, then paths will all point towards whatevers is inside ```project_hypot.sh```. Similarly, you can run ```source set_project.sh stripe``` and the paths will update to all point to the files inside ```project_stripe.sh```. 

Assuming you have done the above, added the files to your ~/.bash_profile we are ready to begin!

Run the following once to start the setup
```bash
source ~/.bash_profile
```


Once this is all setup, and you have added the appropriate lines to your 

### Downloads & Installation 

Installing all the many, many, packages for neuroimaging analyses is a pain. Keeping it consistent, and reusable is even more of a pain. Fortunately we can use a couple of programs to manage the many packages and make installing a little easier

#### [1] mamba/conda (required)
This manages python environments. Basically, it allows you to create install a different python for each project. This is useful because different python programs have different requirements which often get in the way and cause havoc. Mamba is a speedy way to control this. You can also use conda, which will do the same things, but is a little slower. For installation instructions for mamba go to https://github.com/conda-forge/miniforge?tab=readme-ov-file#install 

*If you want to use conda rather than mamba set ```PYPACKAGE_MANAGER=conda``` inside your ```project_*.sh``` file*


#### [2] Docker/Singularity (required)
Docker allows you to make entire virtual machines, like a mini-version of a computer. You can create specific recipes for that machine, which contain all the software you need. Again this avoids trouble with different hardware, allows you to control versions etc. Follow these instructions to install docker: https://docs.docker.com/desktop/setup/install/mac-install/


If you want to overwrite which container manager you use, set the ```export CONTAINER_TYPE="apptainer" # or singularity``` inside your ```project_*.sh``` files 

#### [3] freesurfer+freeview (required)
The pipeline relies on freesurfer for segmentation; which needs to be checked by eye manually. This means it does need to be installed locally, as we need to be able to check surfaces & make edits with the gui. Check the freesurfer version in the config_pipeline.sh file. Currently we are working with 7.3.2. Select the correct version for your system (mac, linux, windows etc)
- https://surfer.nmr.mgh.harvard.edu/pub/dist/freesurfer/7.3.2/
Then follow the instructions here
- https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
You will need to obtain a license key for your email
- https://surfer.nmr.mgh.harvard.edu/registration.html
Save this license file, inside the folder config, in this repository
- /where/you/cloned/this/repo/config/license.txt

#### [4] MRI viewer (optional) *MARCUS CHANGE THIS YOU CAN DO BETTER*
It is also important to have a way to view the images, to check registration, motion, artefacts etc. Everyone will have there own preference. A standard one is to use **fsleyes**. If you don't have it installed already it can be installed with mamba. 
```bash
# We create a new environment with mamba, and install fsleyes to 
$PYPACKAGE_MANAGER create -n fslmamba -c conda-forge -c https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/ fsl-base fsleyes
# 2. Activate the environment
$PYPACKAGE_MANAGER activate fslmamba
# 3. Try opening fsleyes
fsleyes
```
I quite like itksnap; but anything you are used is good. 

#### [5] VS code(optional - highly recommended):
Vscode - use this to edit all of your programming files etc. It is also useful for running notebooks for code. It also can be used to install helpful plugins, like niivue. You could use another IDE if you are used to working in a specific environment. 
- https://code.visualstudio.com/download 
- TODO: add examples of notebooks

#### [6] Everything else...
Once you have this installed you are ready to go. All the other installation steps will be (hopefully!) handelled along the way by either *mamba* or *docker*. Remember for docker commands to work they need to be running  

#### Automatic install of other requirements
Once the above has been installed we then run some stuff to get everything started: 

Assuming you have added  ```source "/path/to/this/folder/config/config_pipeline.sh"``` to your bash profile

Open a new terminal 
```bash
cd "${PIPELINE_DIR}/config"
# Install python environments
bash s00_python_environments.sh
# Install the docker images 
bash s00_containers.sh
```
