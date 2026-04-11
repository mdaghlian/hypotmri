#!/bin/bash
# Make different $PYPACKAGE_MANAGER environments we will use
echo "creating environments using ${PYPACKAGE_MANAGER}"

# ─────────────────────────────────────────────
# Helper: remove + recreate an env
# ─────────────────────────────────────────────
recreate_env() {
    local ENV_NAME="$1"
    if $PYPACKAGE_MANAGER env list | grep -q "^$ENV_NAME "; then
        echo "Environment '$ENV_NAME' found. Removing it..."
        $PYPACKAGE_MANAGER env remove -n $ENV_NAME -y
    else
        echo "Environment '$ENV_NAME' does not exist. Skipping removal."
    fi
}

# ─────────────────────────────────────────────
# Individual install functions
# ─────────────────────────────────────────────
install_b14() {
    local ENV_NAME="b14"
    echo "=== Installing: $ENV_NAME (benson atlas) ==="
    recreate_env $ENV_NAME
    $PYPACKAGE_MANAGER create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install \
        nibabel==$NIBABEL_VERSION \
        neuropythy==$NEUROPYTHY_VERSION \
        nitime
    echo "Done! Activate with: $PYPACKAGE_MANAGER activate $ENV_NAME"
}

install_autoflat() {
    local ENV_NAME="autoflat"
    echo "=== Installing: $ENV_NAME ==="
    recreate_env $ENV_NAME
    $PYPACKAGE_MANAGER create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install autoflatten==$AUTOFLAT_VERSION
    echo "Done! Activate with: $PYPACKAGE_MANAGER activate $ENV_NAME"
}

install_pctx() {
    local ENV_NAME="pctx"
    echo "=== Installing: $ENV_NAME (pycortex) - TRICKY, STILL WORKING ON BEST APPROACH ==="
    conda activate base
    recreate_env $ENV_NAME
    # Has to be 3.10 for pycortex
    $PYPACKAGE_MANAGER create -n $ENV_NAME python=3.10 -y
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install \
        pycortex==$PYCTX_VERSION \
        nibabel==$NIBABEL_VERSION \
        nilearn==$NILEARN_VERSION \
        git+https://github.com/mdaghlian/dpu_mini.git
    echo "Done! Activate with: $PYPACKAGE_MANAGER activate $ENV_NAME"
}

install_preproc() {
    # ************************************************************
    # NOT YET ORGANIZED - considering general purpose python - with fsl
    # fsl is temperamental as a python environment
    # ************************************************************
    local ENV_NAME="preproc"
    echo "=== Installing: $ENV_NAME (general preprocessing: fsl, nibabel, nilearn) ==="
    recreate_env $ENV_NAME

    # Detect platform
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)   FSL_PLATFORM="linux-64" ;;
        Linux-aarch64)  FSL_PLATFORM="linux-aarch64" ;;
        Darwin-x86_64)  FSL_PLATFORM="macos-64" ;;
        Darwin-arm64)   FSL_PLATFORM="macos-M1" ;;
        *) echo "Unsupported platform: $(uname -s)-$(uname -m)"; exit 1 ;;
    esac

    FSL_ENV_URL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fsl-${FSL_VERSION}_${FSL_PLATFORM}.yml"
    curl -L -o fsl_env.yml $FSL_ENV_URL
    $PYPACKAGE_MANAGER env create -n $ENV_NAME --file=fsl_env.yml -y
    rm fsl_env.yml
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install \
        nibabel==$NIBABEL_VERSION \
        nilearn==$NILEARN_VERSION \
        nipype==$NIPYPE_VERSION 
        # antspyx==$ANTS_VERSION
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install -e $PIPELINE_DIR/cvl_utils
    echo "Done! Activate with: $PYPACKAGE_MANAGER activate $ENV_NAME"
}

install_prf() {
    local ENV_NAME="prf"
    echo "=== Installing: $ENV_NAME ==="
    recreate_env $ENV_NAME
    $PYPACKAGE_MANAGER create -n $ENV_NAME python=$PYTHON_FOR_PIPELINE -y
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install \
        nibabel==$NIBABEL_VERSION \
        nilearn==$NILEARN_VERSION
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install git+https://github.com/mdaghlian/prfpy.git
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install git+https://github.com/mdaghlian/dpu_mini.git
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install -e $PIPELINE_DIR/cvl_utils
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install -U setuptools wheel numpy cython
    $PYPACKAGE_MANAGER run -n $ENV_NAME pip install -U pycortex==1.3.0
    echo "Done! Activate with: $PYPACKAGE_MANAGER activate $ENV_NAME"
}

# ─────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────
VALID_ENVS=(b14 autoflat pctx preproc prf)

install_all() {
    for env in "${VALID_ENVS[@]}"; do
        install_$env
    done
}

if [[ $# -eq 0 ]]; then
    install_all
elif [[ "$1" == "--env" ]]; then
    if [[ -z "$2" ]]; then
        echo "Error: --env requires an argument."
        echo "Valid environments: ${VALID_ENVS[*]}"
        exit 1
    fi
    case "$2" in
        b14)      install_b14 ;;
        autoflat) install_autoflat ;;
        pctx)     install_pctx ;;
        preproc)  install_preproc ;;
        prf)      install_prf ;;
        *)
            echo "Error: unknown environment '$2'."
            echo "Valid environments: ${VALID_ENVS[*]}"
            exit 1
            ;;
    esac
else
    echo "Usage: $0 [--env <ENV_NAME>]"
    echo "Valid environments: ${VALID_ENVS[*]}"
    exit 1
fi