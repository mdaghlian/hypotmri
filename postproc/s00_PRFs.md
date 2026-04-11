# Functional pipeline 
First make sure you have the correct project activated 
```bash
source set_project.sh <project-name>
```

## [1] PRF fitting (gauss - prfpy)
```bash
conda activate prf

s01_gauss_prfpy.py \
    --bids-dir $BIDS_DIR \
    --sub hp01 \
    --ses 01 \
    --task pRFLE \
    --input-file s4_conf_sgw-347_po-3_d-0_pca6 \
    --output-file s5_gauss_prfpy_v001 \
    --project hypot --roi b14_V1.

```