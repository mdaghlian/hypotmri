#!/bin/bash
# Make different conda environments we will use
# b14 (benson atlas)
ENV_NAME="b14"
# 2. Check if the directory for the env exists in mamba's info
if mamba env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' found. Removing it..."
    mamba env remove -n $ENV_NAME -y
else
    echo "Environment '$ENV_NAME' does not exist. Skipping removal."
fi
# 3. Create the environment fresh
echo "Creating environment '$ENV_NAME'..."
mamba create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
conda run -n $ENV_NAME pip install nibabel==$NIBABEL_VERSION neuropythy==$NEUROPYTHY_VERSION
echo "Done! You can now activate it using: mamba activate $ENV_NAME"

# # autoflatten
# ENV_NAME="autoflat"
# # 2. Check if the directory for the env exists in mamba's info
# if mamba env list | grep -q "^$ENV_NAME "; then
#     echo "Environment '$ENV_NAME' found. Removing it..."
#     mamba env remove -n $ENV_NAME -y
# else
#     echo "Environment '$ENV_NAME' does not exist. Skipping removal."
# fi
# # 3. Create the environment fresh
# echo "Creating environment '$ENV_NAME'..."
# mamba create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
# conda run -n $ENV_NAME pip install autoflatten==$AUTOFLAT_VERSION
# echo "Done! You can now activate it using: mamba activate $ENV_NAME"
