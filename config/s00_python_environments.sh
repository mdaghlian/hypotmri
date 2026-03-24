#!/bin/bash
# Make different $PYPACKAGE_MANAGER environments we will use
echo "creating environments using ${PYPACKAGE_MANAGER}"
# b14 (benson atlas)
ENV_NAME="b14"
# 2. Check if the directory for the env exists in $PYPACKAGE_MANAGER's info
if $PYPACKAGE_MANAGER env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' found. Removing it..."
    $PYPACKAGE_MANAGER env remove -n $ENV_NAME -y
else
    echo "Environment '$ENV_NAME' does not exist. Skipping removal."
fi
# 3. Create the environment fresh
echo "Creating environment '$ENV_NAME'..."
$PYPACKAGE_MANAGER create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
$PYPACKAGE_MANAGER run -n $ENV_NAME pip install nibabel==$NIBABEL_VERSION neuropythy==$NEUROPYTHY_VERSION
echo "Done! You can now activate it using: $PYPACKAGE_MANAGER activate $ENV_NAME"

# autoflatten
ENV_NAME="autoflat"
# 2. Check if the directory for the env exists in $PYPACKAGE_MANAGER's info
if $PYPACKAGE_MANAGER env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' found. Removing it..."
    $PYPACKAGE_MANAGER env remove -n $ENV_NAME -y
else
    echo "Environment '$ENV_NAME' does not exist. Skipping removal."
fi
# 3. Create the environment fresh
echo "Creating environment '$ENV_NAME'..."
$PYPACKAGE_MANAGER create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
$PYPACKAGE_MANAGER run -n $ENV_NAME pip install autoflatten==$AUTOFLAT_VERSION
echo "Done! You can now activate it using: $PYPACKAGE_MANAGER activate $ENV_NAME"


# Pycortex 
echo "PYCORTEX ENVIRONMENT IS TRICKY - STILL WORKING ON BEST APPROACH" 
conda activate base
ENV_NAME="pctx"
# 2. Check if the directory for the env exists in $PYPACKAGE_MANAGER's info
if $PYPACKAGE_MANAGER env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' found. Removing it..."
    $PYPACKAGE_MANAGER env remove -n $ENV_NAME -y
else
    echo "Environment '$ENV_NAME' does not exist. Skipping removal."
fi
# Has to be 3.10 for pycortex
$PYPACKAGE_MANAGER create -n $ENV_NAME python=3.10 -y
$PYPACKAGE_MANAGER run -n $ENV_NAME pip install pycortex==$PYCTX_VERSION \
    nibabel==$NIBABEL_VERSION \
    nilearn==$NILEARN_VERSION \
    git+https://github.com/mdaghlian/dpu_mini.git 


# ************************************************************
# BELOW IS NOT YET ORGANIZED - considering general purpose python - with fsl
# but fsl is temperamental as a python environment
# ************************************************************



# # PREPROC - general preprocessing: fsl; ants; nibabel; 
# ENV_NAME="preproc"
# # 2. Check if the directory for the env exists in $PYPACKAGE_MANAGER's info
# if $PYPACKAGE_MANAGER env list | grep -q "^$ENV_NAME "; then
#     echo "Environment '$ENV_NAME' found. Removing it..."
#     $PYPACKAGE_MANAGER env remove -n $ENV_NAME -y
# else
#     echo "Environment '$ENV_NAME' does not exist. Skipping removal."
# fi


# # Detect platform
# case "$(uname -s)-$(uname -m)" in
#     Linux-x86_64)   FSL_PLATFORM="linux-64" ;;
#     Linux-aarch64)  FSL_PLATFORM="linux-aarch64" ;;
#     Darwin-x86_64)  FSL_PLATFORM="macos-64" ;;
#     Darwin-arm64)   FSL_PLATFORM="macos-M1" ;;
#     *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
# esac

# FSL_ENV_URL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fsl-${FSL_VERSION}_${FSL_PLATFORM}.yml"

# $PYPACKAGE_MANAGER env create \
#   -n $ENV_NAME \
#   -f $FSL_ENV_URL

# $PYPACKAGE_MANAGER run -n $ENV_NAME pip install \
#     antspyx==$ANTS_VERSION \
#     nibabel==$NIBABEL_VERSION \
#     nilearn==$NILEARN_VERSION


