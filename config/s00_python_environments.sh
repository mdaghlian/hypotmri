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
